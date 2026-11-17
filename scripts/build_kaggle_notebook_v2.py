#!/usr/bin/env python3
"""
Builds the Kaggle training notebook for the NeuroFoil surrogate (v2).

v2 changes vs the legacy notebook:
- Grouped (airfoil-level) train/val/test splits: no geometry leakage between
  splits; interpolation vs shape-extrapolation reported separately.
- All RNG seeds fixed; deterministic training.
- Field decoder predicts BOTH surfaces (322 outputs: cp_upper 161 then
  cp_lower 161) so the runtime can plot each surface.
- Split-conformal calibration (95% quantiles of held-out absolute residuals)
  for scalars and per-chord-point Cp, with empirical coverage reported.
- Exports models/surrogate_weights.npz + surrogate_meta.json for the pure
  NumPy runtime in src/models/inference.py, plus .pt checkpoints and scalers.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "neurofoil_surrogate_training_v2.ipynb"

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": source.splitlines(keepends=True),
    })


md("""# NeuroFoil Surrogate Training (v2) — Leakage-Free Splits + Conformal UQ

Trains the aerodynamic surrogate on the **real OpenFOAM CFD campaign dataset**:

1. **Deep Residual MLP** (PyTorch) → scalar coefficients $(C_L, C_D, C_m)$
2. **1D Field Decoder** (PyTorch) → surface pressure $C_p(x)$ on both surfaces (322 outputs)
3. **Split-conformal calibration** → 95% intervals with measured coverage

**Data hygiene (v2):** train/val/test are split **by airfoil** (grouped split), so the
test set contains *never-seen geometries* — shape extrapolation, not memorization.
A random-row split is also trained for the interpolation-vs-extrapolation comparison.

**Setup:** attach the dataset containing `airfoil_cfd_dataset.csv` +
`cp_surface_distributions.npz` (Kaggle sidebar → Input). GPU accelerator is optional;
the models are small.""")

code("""# [1] Environment & seeding
import os, sys, json, time, random, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | device: {device}")""")

code("""# [2] Locate and load the CFD campaign dataset
def find_data():
    roots = [Path('/kaggle/input'), Path('.'), Path('data/processed'), Path('../data/processed'), Path('/content')]
    hits = []
    for root in roots:
        if root.exists():
            hits += list(root.rglob('airfoil_cfd_dataset.csv'))
    if not hits:
        raise FileNotFoundError('airfoil_cfd_dataset.csv not found — attach the dataset in the Input panel.')
    csv_path = hits[0]
    npz_path = csv_path.parent / 'cp_surface_distributions.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'cp_surface_distributions.npz not found next to {csv_path}')
    return csv_path, npz_path

csv_path, npz_path = find_data()
print(f'CSV: {csv_path}')
print(f'NPZ: {npz_path}')

df_raw = pd.read_csv(csv_path)
npz = np.load(npz_path, allow_pickle=False)
print(f'Rows: {len(df_raw)} | columns: {len(df_raw.columns)}')""")

code("""# [3] Validate and align features / targets (order must match src/models/inference.py)
FEATURE_ORDER = (
    ['m', 'p', 't']
    + [f'cst_u_{i}' for i in range(6)]
    + [f'cst_l_{i}' for i in range(6)]
    + ['alpha_deg', 'reynolds']
)
TARGET_ORDER = ['cl', 'cd', 'cm']

required_columns = set(
    FEATURE_ORDER + TARGET_ORDER + ['status', 'naca_code', 'case_id', 'data_source']
)
missing_columns = sorted(required_columns.difference(df_raw.columns))
if missing_columns:
    raise ValueError(f'CSV is missing required columns: {missing_columns}')

converged_mask = df_raw['status'].eq('converged')
if not converged_mask.any():
    raise ValueError('CSV contains no converged OpenFOAM rows for training.')
converged_sources = df_raw.loc[converged_mask, 'data_source']
if converged_sources.isna().any() or not converged_sources.eq('openfoam').all():
    bad_sources = sorted(converged_sources.fillna('<missing>').astype(str).unique())
    raise ValueError(
        'All converged training rows must have data_source="openfoam"; '
        f'found {bad_sources}. Refusing legacy, unlabeled, or non-OpenFOAM data.'
    )
if 'is_synthetic' in df_raw.columns:
    synthetic_flags = df_raw['is_synthetic']
    if synthetic_flags.isna().any():
        raise ValueError('is_synthetic must be non-null when the column is present.')
    if synthetic_flags.dtype == bool:
        synthetic_mask = synthetic_flags
    elif pd.api.types.is_numeric_dtype(synthetic_flags):
        if not synthetic_flags.isin([0, 1]).all():
            raise ValueError('is_synthetic must contain only boolean or 0/1 values.')
        synthetic_mask = synthetic_flags.astype(bool)
    else:
        normalized_synthetic = synthetic_flags.astype(str).str.strip().str.lower()
        if not normalized_synthetic.isin(['true', 'false', '0', '1']).all():
            raise ValueError('is_synthetic must contain only boolean or 0/1 values.')
        synthetic_mask = normalized_synthetic.isin(['true', '1'])
    if synthetic_mask.any():
        raise ValueError(
            f'Found {int(synthetic_mask.sum())} synthetic rows; refusing to train the canonical OpenFOAM model.'
        )

required_npz_arrays = {'x_grid', 'case_ids', 'cp_upper', 'cp_lower'}
missing_npz_arrays = sorted(required_npz_arrays.difference(npz.files))
if missing_npz_arrays:
    raise ValueError(f'Cp NPZ is missing required arrays: {missing_npz_arrays}')

x_grid = np.asarray(npz['x_grid'])
npz_case_ids_raw = np.asarray(npz['case_ids'])
cp_upper_npz = np.asarray(npz['cp_upper'])
cp_lower_npz = np.asarray(npz['cp_lower'])
if x_grid.ndim != 1 or x_grid.size == 0:
    raise ValueError(f'x_grid must be a nonempty 1D array; got {x_grid.shape}')
if npz_case_ids_raw.ndim != 1 or npz_case_ids_raw.size == 0:
    raise ValueError(f'case_ids must be a nonempty 1D array; got {npz_case_ids_raw.shape}')
expected_field_shape = (npz_case_ids_raw.size, x_grid.size)
if cp_upper_npz.shape != expected_field_shape or cp_lower_npz.shape != expected_field_shape:
    raise ValueError(
        'Cp arrays must align with case_ids and x_grid: '
        f'expected {expected_field_shape}, got upper={cp_upper_npz.shape}, lower={cp_lower_npz.shape}'
    )

def normalize_case_id(value):
    return value.decode('utf-8') if isinstance(value, (bytes, np.bytes_)) else str(value)

npz_case_ids = [normalize_case_id(cid) for cid in npz_case_ids_raw]
csv_case_ids = df_raw['case_id'].map(normalize_case_id)
if len(set(npz_case_ids)) != len(npz_case_ids):
    raise ValueError('Cp NPZ case_ids must be unique.')
if csv_case_ids.duplicated().any():
    raise ValueError('CSV case_id values must be unique for unambiguous Cp alignment.')
csv_case_id_set = set(csv_case_ids)
unknown_case_ids = sorted(set(npz_case_ids).difference(csv_case_id_set))
if unknown_case_ids:
    raise ValueError(f'Cp NPZ contains case_ids absent from the CSV: {unknown_case_ids[:5]}')

print(f"Reynolds levels: {sorted(df_raw['reynolds'].unique())}")
print(f"Distinct airfoils: {df_raw['naca_code'].nunique()}")
print(f"Status counts:\\n{df_raw['status'].value_counts().to_string()}")

# Keep honestly-converged, provenance-verified OpenFOAM cases only
df = df_raw[converged_mask].reset_index(drop=True)
print(f'\\nConverged rows used: {len(df)} (dropped {len(df_raw) - len(df)})')

N_PTS = len(x_grid)                       # 161 chordwise points
N_OUT_FIELD = 2 * N_PTS                   # cp_upper(161) then cp_lower(161)

X = df[FEATURE_ORDER].values.astype(np.float32)
y_scalars = df[TARGET_ORDER].values.astype(np.float32)

# Align Cp fields to dataframe rows by unique case_id
df_case_ids = df['case_id'].map(normalize_case_id).tolist()
cp_index = {cid: i for i, cid in enumerate(npz_case_ids)}
rows_with_cp = [i for i, cid in enumerate(df_case_ids) if cid in cp_index]
cp_upper = np.zeros((len(df), N_PTS), dtype=np.float32)
cp_lower = np.zeros((len(df), N_PTS), dtype=np.float32)
has_cp = np.zeros(len(df), dtype=bool)
for i in rows_with_cp:
    j = cp_index[df_case_ids[i]]
    cp_upper[i] = cp_upper_npz[j]
    cp_lower[i] = cp_lower_npz[j]
    has_cp[i] = True
y_fields = np.concatenate([cp_upper, cp_lower], axis=1)

print(f'X: {X.shape} | y_scalars: {y_scalars.shape} | y_fields: {y_fields.shape}')
print(f'Rows with Cp field: {has_cp.sum()}/{len(df)}')""")

code("""# [4] Grouped (airfoil-level) split + random-row reference split
groups = df['naca_code'].values
unique_codes = sorted(df['naca_code'].unique())
if len(unique_codes) < 6:
    raise ValueError(
        'Grouped train/validation/test splitting requires at least 6 distinct '
        f'airfoils; found {len(unique_codes)}.'
    )
rng = np.random.default_rng(SEED)
rng.shuffle(unique_codes)

n_test = max(2, int(round(0.15 * len(unique_codes))))
n_val = max(2, int(round(0.15 * len(unique_codes))))
if n_test + n_val >= len(unique_codes):
    raise ValueError('Grouped split leaves no airfoils for training.')
test_codes = set(unique_codes[:n_test])
val_codes = set(unique_codes[n_test:n_test + n_val])
train_codes = set(unique_codes[n_test + n_val:])

is_test = np.isin(groups, list(test_codes))
is_val = np.isin(groups, list(val_codes))
is_train = ~(is_test | is_val)
if not (is_train.any() and is_val.any() and is_test.any()):
    raise ValueError('Grouped train/validation/test row splits must all be nonempty.')

mask_tr_cp = is_train & has_cp
mask_va_cp = is_val & has_cp
mask_te_cp = is_test & has_cp
if not (mask_tr_cp.any() and mask_va_cp.any() and mask_te_cp.any()):
    raise ValueError(
        'Each grouped split must contain at least one row with an aligned Cp field; '
        f'got train={mask_tr_cp.sum()}, val={mask_va_cp.sum()}, test={mask_te_cp.sum()}.'
    )

print(f'Airfoils: train={len(train_codes)}, val={len(val_codes)}, test={len(test_codes)}')
print(f'Rows:     train={is_train.sum()}, val={is_val.sum()}, test={is_test.sum()}')
print(f'Cp rows:  train={mask_tr_cp.sum()}, val={mask_va_cp.sum()}, test={mask_te_cp.sum()}')
print(f'Test airfoils (never seen in training): {sorted(test_codes)}')

# Random-row split (interpolation reference) — same overall proportions
idx_all = np.arange(len(df))
idx_trv, idx_rnd_test = train_test_split(idx_all, test_size=0.15, random_state=SEED)
idx_rnd_train, idx_rnd_val = train_test_split(idx_trv, test_size=0.1765, random_state=SEED)""")

code("""# [5] Model definitions (must mirror src/models/inference.py numpy forward pass)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class AeroSurrogateMLP(nn.Module):
    def __init__(self, in_features, out_features=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, 128), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, out_features))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)

class AeroFieldDecoder(nn.Module):
    def __init__(self, in_features, n_out=322):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, n_out),
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, X_tr, y_tr, X_va, y_va, epochs=300, batch=32, lr=1e-3, wd=1e-4):
    tr = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    best_val, best_state = float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in va:
                vloss += crit(model(bx.to(device)), by.to(device)).item() * len(bx)
        vloss /= max(len(va.dataset), 1)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val

def regression_report(y_true, y_pred, labels):
    rows = []
    for i, name in enumerate(labels):
        rows.append({
            'target': name,
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
            'RMSE': float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
        })
    return pd.DataFrame(rows)""")

code("""# [6] Train scalar surrogate — grouped split (shape extrapolation)
scaler_x_g = StandardScaler().fit(X[is_train])
scaler_y_g = StandardScaler().fit(y_scalars[is_train])

Xtr = scaler_x_g.transform(X[is_train]); ytr = scaler_y_g.transform(y_scalars[is_train])
Xva = scaler_x_g.transform(X[is_val]);   yva = scaler_y_g.transform(y_scalars[is_val])
Xte = scaler_x_g.transform(X[is_test]);  yte = y_scalars[is_test]

torch.manual_seed(SEED)
mlp_grouped = AeroSurrogateMLP(len(FEATURE_ORDER), 3).to(device)
mlp_grouped, best_val = train_model(mlp_grouped, Xtr, ytr, Xva, yva)
print(f'Best val MSE (normalized): {best_val:.5f}')

mlp_grouped.eval()
with torch.no_grad():
    pred_te = scaler_y_g.inverse_transform(mlp_grouped(torch.tensor(Xte).to(device)).cpu().numpy())
report_grouped = regression_report(yte, pred_te, TARGET_ORDER)
print('GROUPED SPLIT (unseen airfoils) — test metrics:')
print(report_grouped.to_string(index=False))""")

code("""# [7] Train scalar surrogate — random-row split (interpolation reference)
scaler_x_r = StandardScaler().fit(X[idx_rnd_train])
scaler_y_r = StandardScaler().fit(y_scalars[idx_rnd_train])
Xtr_r = scaler_x_r.transform(X[idx_rnd_train]); ytr_r = scaler_y_r.transform(y_scalars[idx_rnd_train])
Xva_r = scaler_x_r.transform(X[idx_rnd_val]);   yva_r = scaler_y_r.transform(y_scalars[idx_rnd_val])
Xte_r = scaler_x_r.transform(X[idx_rnd_test]);  yte_r = y_scalars[idx_rnd_test]

torch.manual_seed(SEED)
mlp_random = AeroSurrogateMLP(len(FEATURE_ORDER), 3).to(device)
mlp_random, _ = train_model(mlp_random, Xtr_r, ytr_r, Xva_r, yva_r)
mlp_random.eval()
with torch.no_grad():
    pred_te_r = scaler_y_r.inverse_transform(mlp_random(torch.tensor(Xte_r).to(device)).cpu().numpy())
report_random = regression_report(yte_r, pred_te_r, TARGET_ORDER)
print('RANDOM-ROW SPLIT (interpolation) — test metrics:')
print(report_random.to_string(index=False))

print('\\nLeakage effect (R2 drop from interpolation to extrapolation):')
for i, name in enumerate(TARGET_ORDER):
    print(f"  {name}: {report_random['R2'][i]:.4f} -> {report_grouped['R2'][i]:.4f}")""")

code("""# [8] Train field decoder (grouped split) + physics-consistency check
# Runtime standardizes every model input with scaler_x_g, so field training and
# every field evaluation path must use that same input representation.
Xtr_field = scaler_x_g.transform(X[mask_tr_cp])
Xva_field = scaler_x_g.transform(X[mask_va_cp])
Xte_field = scaler_x_g.transform(X[mask_te_cp])

field_g = StandardScaler().fit(y_fields[mask_tr_cp])
ytr_field = field_g.transform(y_fields[mask_tr_cp])
yva_field = field_g.transform(y_fields[mask_va_cp])

if Xtr_field.shape[0] != ytr_field.shape[0] or Xva_field.shape[0] != yva_field.shape[0]:
    raise ValueError('Field feature/target rows are misaligned after grouped splitting.')

torch.manual_seed(SEED)
field_model = AeroFieldDecoder(len(FEATURE_ORDER), N_OUT_FIELD).to(device)
field_model, f_best = train_model(
    field_model, Xtr_field, ytr_field, Xva_field, yva_field, epochs=300
)
print(f'Best val MSE (normalized): {f_best:.5f}')

field_model.eval()
with torch.no_grad():
    pred_f = field_g.inverse_transform(
        field_model(torch.tensor(Xte_field).to(device)).cpu().numpy()
    )
true_f = y_fields[mask_te_cp]

r2_flat = r2_score(true_f.flatten(), pred_f.flatten())
rmse_flat = float(np.sqrt(mean_squared_error(true_f.flatten(), pred_f.flatten())))
print(f'Cp field (both surfaces) — R2: {r2_flat:.4f} | RMSE: {rmse_flat:.5f}')

# Physics consistency: integrate delta-Cp lift vs true CL
d_pred = pred_f[:, N_PTS:] - pred_f[:, :N_PTS]
d_true = true_f[:, N_PTS:] - true_f[:, :N_PTS]
cl_int_pred = np.trapezoid(d_pred, x_grid, axis=1)
cl_int_true = np.trapezoid(d_true, x_grid, axis=1)
r2_phys = r2_score(cl_int_true, cl_int_pred)
print(f'Physics consistency: R2(integrated dCp lift vs true CL) = {r2_phys:.4f}')""")

code("""# [9] Split-conformal calibration (95% intervals) + coverage audit
with torch.no_grad():
    val_pred = scaler_y_g.inverse_transform(mlp_grouped(torch.tensor(Xva).to(device)).cpu().numpy())

res_val = np.abs(y_scalars[is_val] - val_pred)
n_cal = len(res_val)

def conformal_order_statistic(abs_residuals, coverage=0.95):
    values = np.asarray(abs_residuals)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError('Conformal calibration requires at least one residual row.')
    rank = min(math.ceil((values.shape[0] + 1) * coverage), values.shape[0])
    return np.sort(values, axis=0)[rank - 1]

q_scalars = conformal_order_statistic(res_val)
q_cl, q_cd, q_cm = (float(value) for value in q_scalars)

# Per-chord-point conformal quantiles use their own Cp calibration count.
Xcal_field = scaler_x_g.transform(X[mask_va_cp])
with torch.no_grad():
    val_field_pred = field_g.inverse_transform(
        field_model(torch.tensor(Xcal_field).to(device)).cpu().numpy()
    )
res_field_val = np.abs(y_fields[mask_va_cp] - val_field_pred)
n_field_cal = len(res_field_val)
q_cp = conformal_order_statistic(res_field_val)
q_cp_upper = q_cp[:N_PTS]
q_cp_lower = q_cp[N_PTS:]

# Empirical coverage on the untouched grouped test set
cov_cl = float(np.mean(np.abs(yte[:, 0] - pred_te[:, 0]) <= q_cl))
cov_cd = float(np.mean(np.abs(yte[:, 1] - pred_te[:, 1]) <= q_cd))
cov_cm = float(np.mean(np.abs(yte[:, 2] - pred_te[:, 2]) <= q_cm))
covered_cp_upper = np.abs(true_f[:, :N_PTS] - pred_f[:, :N_PTS]) <= q_cp_upper
covered_cp_lower = np.abs(true_f[:, N_PTS:] - pred_f[:, N_PTS:]) <= q_cp_lower
cov_cp_upper = float(np.mean(covered_cp_upper))
cov_cp_lower = float(np.mean(covered_cp_lower))
cov_cp_combined = float(np.mean(np.concatenate([covered_cp_upper, covered_cp_lower], axis=1)))
print(f'q_CL={q_cl:.4f}  q_CD={q_cd:.5f}  q_Cm={q_cm:.4f}')
print(
    f'Test coverage: CL {cov_cl:.3f} | CD {cov_cd:.3f} | Cm {cov_cm:.3f} | '
    f'Cp upper {cov_cp_upper:.3f} | Cp lower {cov_cp_lower:.3f} | '
    f'Cp combined {cov_cp_combined:.3f}  (nominal target 0.95)'
)""")

code("""# [10] OOD sanity check: Mahalanobis distance for unseen airfoils vs seen
from scipy.stats import chi2
# Work in the same standardized feature space used by both neural decoders;
# otherwise Reynolds units dominate geometry/CST dimensions in the covariance.
feat_tr_norm = scaler_x_g.transform(X[is_train])
mu_norm = feat_tr_norm.mean(axis=0)
cov_norm = np.cov(feat_tr_norm, rowvar=False) + np.eye(len(FEATURE_ORDER)) * 1e-6
inv_norm = np.linalg.inv(cov_norm)

def mahal_normalized(Xq):
    Xq_norm = scaler_x_g.transform(Xq)
    d = Xq_norm - mu_norm
    return np.sqrt(np.einsum('ij,jk,ik->i', d, inv_norm, d))

d_train, d_test = mahal_normalized(X[is_train]), mahal_normalized(X[is_test])
thr = float(np.sqrt(chi2.ppf(0.99, df=len(FEATURE_ORDER))))
print(f'Mahalanobis: seen-airfoil rows median={np.median(d_train):.2f}, p99={np.percentile(d_train, 99):.2f}')
print(f'             unseen-airfoil rows median={np.median(d_test):.2f}, max={d_test.max():.2f}')
print(f'chi2(0.99) threshold: {thr:.2f} | unseen rows beyond threshold: {(d_test > thr).mean():.1%}')""")

code("""# [11] Export artifacts for the NumPy runtime + paper
out_dir = Path('models'); out_dir.mkdir(exist_ok=True)

# 1. Canonical PyTorch checkpoints
torch.save(mlp_grouped.state_dict(), out_dir / 'best_surrogate_scalars.pt')
torch.save(field_model.state_dict(), out_dir / 'best_surrogate_cp.pt')

# 2. Scalers (consumed by src/models/inference.py)
np.save(out_dir / 'scaler_x_mean.npy', scaler_x_g.mean_)
np.save(out_dir / 'scaler_x_scale.npy', scaler_x_g.scale_)
np.save(out_dir / 'scaler_y_mean.npy', scaler_y_g.mean_)
np.save(out_dir / 'scaler_y_scale.npy', scaler_y_g.scale_)
np.save(out_dir / 'scaler_field_mean.npy', field_g.mean_)
np.save(out_dir / 'scaler_field_scale.npy', field_g.scale_)

# 3. NumPy-friendly weights (consumed by the pure-NumPy runtime)
sd_mlp = mlp_grouped.state_dict()
sd_fld = field_model.state_dict()
W = {
    's_in_w': sd_mlp['input_layer.0.weight'].numpy(), 's_in_b': sd_mlp['input_layer.0.bias'].numpy(),
    's_in_bn_m': sd_mlp['input_layer.1.running_mean'].numpy(), 's_in_bn_v': sd_mlp['input_layer.1.running_var'].numpy(),
    's_in_bn_g': sd_mlp['input_layer.1.weight'].numpy(), 's_in_bn_b': sd_mlp['input_layer.1.bias'].numpy(),
}
for r, prefix in ((1, 'res1.'), (2, 'res2.')):
    W.update({
        f's_r{r}_fc_w': sd_mlp[prefix + 'net.0.weight'].numpy(), f's_r{r}_fc_b': sd_mlp[prefix + 'net.0.bias'].numpy(),
        f's_r{r}_bn1_m': sd_mlp[prefix + 'net.1.running_mean'].numpy(), f's_r{r}_bn1_v': sd_mlp[prefix + 'net.1.running_var'].numpy(),
        f's_r{r}_bn1_g': sd_mlp[prefix + 'net.1.weight'].numpy(), f's_r{r}_bn1_b': sd_mlp[prefix + 'net.1.bias'].numpy(),
        f's_r{r}_fc2_w': sd_mlp[prefix + 'net.3.weight'].numpy(), f's_r{r}_fc2_b': sd_mlp[prefix + 'net.3.bias'].numpy(),
        f's_r{r}_bn2_m': sd_mlp[prefix + 'net.4.running_mean'].numpy(), f's_r{r}_bn2_v': sd_mlp[prefix + 'net.4.running_var'].numpy(),
        f's_r{r}_bn2_g': sd_mlp[prefix + 'net.4.weight'].numpy(), f's_r{r}_bn2_b': sd_mlp[prefix + 'net.4.bias'].numpy(),
    })
W.update({
    's_h1_w': sd_mlp['head.0.weight'].numpy(), 's_h1_b': sd_mlp['head.0.bias'].numpy(),
    's_out_w': sd_mlp['head.2.weight'].numpy(), 's_out_b': sd_mlp['head.2.bias'].numpy(),
    'f_l1_w': sd_fld['net.0.weight'].numpy(), 'f_l1_b': sd_fld['net.0.bias'].numpy(),
    'f_l2_w': sd_fld['net.2.weight'].numpy(), 'f_l2_b': sd_fld['net.2.bias'].numpy(),
    'f_l3_w': sd_fld['net.4.weight'].numpy(), 'f_l3_b': sd_fld['net.4.bias'].numpy(),
    'f_l4_w': sd_fld['net.6.weight'].numpy(), 'f_l4_b': sd_fld['net.6.bias'].numpy(),
})
np.savez_compressed(out_dir / 'surrogate_weights.npz', **W)

# 4. Metadata: feature order, conformal params, bounds, metrics
meta = {
    'feature_order': FEATURE_ORDER,
    'target_order': TARGET_ORDER,
    'data_source': 'openfoam',
    'is_synthetic': False,
    'source_row_count': int(len(df)),
    'source_status_counts': {
        str(status): int(count) for status, count in df_raw['status'].value_counts().items()
    },
    'has_field_scaler': True,
    'field_input_scaler': 'scaler_x',
    'field_target_scaler': 'scaler_field',
    'arch': {'scalars': 'resmlp_128x2_head64', 'field': 'mlp_128_256_512_322',
             'field_output': 'cp_upper_161_then_cp_lower_161'},
    'x_grid': x_grid.astype(float).tolist(),
    'conformal': {
        'method': 'split_conformal_abs_residual_95pct',
        'calibration_n': int(n_cal),
        'field_calibration_n': int(n_field_cal),
        'q_cl': q_cl, 'q_cd': q_cd, 'q_cm': q_cm,
        'q_cp_upper': q_cp_upper.tolist(), 'q_cp_lower': q_cp_lower.tolist(),
    },
    'bounds': {'m': [0.0, 0.06], 'p': [0.0, 0.60], 't': [0.06, 0.18],
               'alpha_deg': [-6.0, 14.0], 'reynolds': [1.0e6, 6.0e6]},
    'metrics': {
        'grouped_test': {name: {'R2': float(report_grouped['R2'][i]),
                                'RMSE': float(report_grouped['RMSE'][i]),
                                'MAE': float(report_grouped['MAE'][i])}
                         for i, name in enumerate(TARGET_ORDER)},
        'random_test': {name: {'R2': float(report_random['R2'][i]),
                               'RMSE': float(report_random['RMSE'][i]),
                               'MAE': float(report_random['MAE'][i])}
                        for i, name in enumerate(TARGET_ORDER)},
        'field_grouped_test': {'R2_flat': float(r2_flat), 'RMSE_flat': rmse_flat,
                               'R2_integrated_CL': float(r2_phys)},
        'coverage_test': {
            'cl': cov_cl,
            'cd': cov_cd,
            'cm': cov_cm,
            'cp_upper_pointwise': cov_cp_upper,
            'cp_lower_pointwise': cov_cp_lower,
            'cp_combined_pointwise': cov_cp_combined,
        },
        'n_train_rows': int(is_train.sum()), 'n_val_rows': int(is_val.sum()),
        'n_test_rows': int(is_test.sum()),
        'n_airfoils': int(len(unique_codes)),
        'reynolds_levels': sorted(float(v) for v in df['reynolds'].unique()),
    },
    'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'seed': SEED,
}
(out_dir / 'surrogate_meta.json').write_text(json.dumps(meta, indent=2))

# 5. Bundle for download
with zipfile.ZipFile('neurofoil_models.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in out_dir.iterdir():
        z.write(f, f.name)
print('Exported:', sorted(p.name for p in out_dir.iterdir()))
print('Download neurofoil_models.zip and extract into the repository models/ folder.')""")

md("""## Next steps

1. Download **`neurofoil_models.zip`** (right panel → Output).
2. Extract into the repository `models/` folder — it contains
   `surrogate_weights.npz`, `surrogate_meta.json`, the `.pt` checkpoints, and the
   input, scalar-target, and field-target scaler arrays.
3. The Streamlit app and `AeroSurrogateEngine` now run the **trained neural
   network** (pure NumPy, no PyTorch needed at runtime) with conformal 95%
   intervals; `predict(...)['model_backend']` will read `neural`.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")

