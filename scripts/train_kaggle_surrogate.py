"""
Kaggle / Cloud PyTorch Training Script for Aerodynamic Surrogate Models.

Trains:
1. Multi-Task Deep Residual MLP for scalar coefficients (CL, CD, Cm).
2. 1D Physics-Informed Decoder for continuous surface pressure Cp(x).
3. XGBoost regressor baseline for benchmark comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def find_dataset_files() -> tuple[Path, Path]:
    """Find dataset files across local, repo, or Kaggle input paths."""
    candidates = [
        # Kaggle input paths
        Path("/kaggle/input/airfoil-cfd-dataset"),
        Path("/kaggle/input/ml-aerodynamic-design-platform"),
        Path("/kaggle/input/airfoil-dataset"),
        # Local paths
        Path("data/processed"),
        Path("../data/processed"),
        Path("../../data/processed"),
        Path("."),
    ]
    for c in candidates:
        csv_file = c / "airfoil_cfd_dataset.csv"
        npz_file = c / "cp_surface_distributions.npz"
        if csv_file.exists() and npz_file.exists():
            return csv_file, npz_file

    raise FileNotFoundError("Could not locate airfoil_cfd_dataset.csv and cp_surface_distributions.npz")


def main() -> None:
    print("=" * 72)
    print("AERODYNAMIC SCI-ML SURROGATE MODEL TRAINING (KAGGLE / CLOUD PIPELINE)")
    print("=" * 72)

    csv_path, npz_path = find_dataset_files()
    print(f"Loading CSV data from : {csv_path}")
    print(f"Loading NPZ data from : {npz_path}")

    df = pd.read_csv(csv_path)
    npz = np.load(npz_path)

    x_grid = npz["x_grid"]
    cp_upper = npz["cp_upper"]
    cp_lower = npz["cp_lower"]
    delta_cp = npz["delta_cp"]

    print(f"Dataset size: {len(df)} samples across {df['naca_code'].nunique()} airfoil shapes")

    # Feature definitions
    cst_cols = [c for c in df.columns if c.startswith("cst_")]
    geom_cols = ["m", "p", "t"] + cst_cols
    flow_cols = ["alpha_deg", "reynolds"]
    feature_cols = geom_cols + flow_cols
    target_cols = ["cl", "cd", "cm"]

    print(f"Features ({len(feature_cols)}): {feature_cols[:6]} ... {feature_cols[-4:]}")
    print(f"Targets ({len(target_cols)}): {target_cols}")

    X = df[feature_cols].values
    y_scalars = df[target_cols].values
    y_fields = delta_cp  # (N, 161) surface pressure difference

    # Split train/val/test (80/10/10)
    indices = np.arange(len(df))
    idx_train_val, idx_test = train_test_split(indices, test_size=0.10, random_state=42)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.1111, random_state=42)

    print(f"Split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    # Scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X[idx_train])
    X_val = scaler_x.transform(X[idx_val])
    X_test = scaler_x.transform(X[idx_test])

    y_train = scaler_y.fit_transform(y_scalars[idx_train])
    y_val = scaler_y.transform(y_scalars[idx_val])
    y_test = scaler_y.transform(y_scalars[idx_test])

    y_fields_train = y_fields[idx_train]
    y_fields_val = y_fields[idx_val]
    y_fields_test = y_fields[idx_test]

    # Check for PyTorch availability
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[PyTorch Setup] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        # -------------------------------------------------------------
        # Model 1: Deep Residual MLP for Scalar Coefficients (CL, CD, Cm)
        # -------------------------------------------------------------
        class ResidualBlock(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(x + self.net(x))

        class AeroSurrogateMLP(nn.Module):
            def __init__(self, in_features: int, out_features: int = 3):
                super().__init__()
                self.input_layer = nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                )
                self.res1 = ResidualBlock(128)
                self.res2 = ResidualBlock(128)
                self.head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, out_features),
                )

            def forward(self, x):
                x = self.input_layer(x)
                x = self.res1(x)
                x = self.res2(x)
                return self.head(x)

        # DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        mlp_model = AeroSurrogateMLP(in_features=X.shape[1], out_features=3).to(device)
        optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.MSELoss()

        print("\nTraining Model 1 (Deep Residual MLP) for 100 epochs...")
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, 101):
            mlp_model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = mlp_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(train_dataset)
            scheduler.step()

            mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = mlp_model(bx)
                    val_loss += criterion(pred, by).item() * len(bx)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in mlp_model.state_dict().items()}

            if epoch % 20 == 0 or epoch == 100:
                print(f"  Epoch {epoch:03d}/100 | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if best_state is not None:
            mlp_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Test evaluation
        mlp_model.eval()
        with torch.no_grad():
            preds_norm = mlp_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        preds_scalars = scaler_y.inverse_transform(preds_norm)
        y_test_scalars = y_scalars[idx_test]

        print("\n" + "=" * 54)
        print("MODEL 1 (DEEP RESIDUAL MLP) TEST EVALUATION")
        print("=" * 54)
        for i, target in enumerate(target_cols):
            r2 = r2_score(y_test_scalars[:, i], preds_scalars[:, i])
            rmse = np.sqrt(mean_squared_error(y_test_scalars[:, i], preds_scalars[:, i]))
            mae = mean_absolute_error(y_test_scalars[:, i], preds_scalars[:, i])
            print(f"  {target.upper():<4} -> R² = {r2:.4f} | RMSE = {rmse:.5e} | MAE = {mae:.5e}")
        print("=" * 54)

        # -------------------------------------------------------------
        # Model 2: 1D Continuous Surface Pressure Cp(x) Field Predictor
        # -------------------------------------------------------------
        class AeroFieldDecoder(nn.Module):
            """Predicts 161 continuous chordwise delta Cp(x) values from geometry and alpha."""

            def __init__(self, in_features: int, n_pts: int = 161):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.GELU(),
                    nn.Linear(128, 256),
                    nn.GELU(),
                    nn.Linear(256, 512),
                    nn.GELU(),
                    nn.Linear(512, n_pts),
                )

            def forward(self, x):
                return self.net(x)

        field_model = AeroFieldDecoder(in_features=X.shape[1], n_pts=161).to(device)
        field_opt = torch.optim.AdamW(field_model.parameters(), lr=1e-3, weight_decay=1e-5)
        field_sched = torch.optim.lr_scheduler.CosineAnnealingLR(field_opt, T_max=100)

        t_f_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_fields_train, dtype=torch.float32))
        t_f_val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_fields_val, dtype=torch.float32))
        loader_f_train = DataLoader(t_f_train, batch_size=32, shuffle=True)
        loader_f_val = DataLoader(t_f_val, batch_size=64, shuffle=False)

        print("\nTraining Model 2 (Surface Cp Field Decoder) for 100 epochs...")
        best_field_val = float("inf")
        best_field_state = None

        for epoch in range(1, 101):
            field_model.train()
            tr_loss = 0.0
            for bx, by in loader_f_train:
                bx, by = bx.to(device), by.to(device)
                field_opt.zero_grad()
                pred = field_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                field_opt.step()
                tr_loss += loss.item() * len(bx)
            tr_loss /= len(t_f_train)
            field_sched.step()

            field_model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for bx, by in loader_f_val:
                    bx, by = bx.to(device), by.to(device)
                    pred = field_model(bx)
                    va_loss += criterion(pred, by).item() * len(bx)
            va_loss /= len(t_f_val)

            if va_loss < best_field_val:
                best_field_val = va_loss
                best_field_state = {k: v.cpu() for k, v in field_model.state_dict().items()}

            if epoch % 20 == 0 or epoch == 100:
                print(f"  Epoch {epoch:03d}/100 | Field Train MSE: {tr_loss:.5f} | Val MSE: {va_loss:.5f}")

        if best_field_state is not None:
            field_model.load_state_dict({k: v.to(device) for k, v in best_field_state.items()})

        # Field evaluation
        field_model.eval()
        with torch.no_grad():
            pred_fields = field_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

        r2_field = r2_score(y_fields_test.flatten(), pred_fields.flatten())
        rmse_field = np.sqrt(mean_squared_error(y_fields_test.flatten(), pred_fields.flatten()))
        print(f"\nSurface Pressure Delta Cp(x) Overall -> R² = {r2_field:.4f} | RMSE = {rmse_field:.5e}")

        # Check integrated lift consistency: CL_integrated = int_0^1 (Delta_Cp) dx
        cl_integrated = np.trapezoid(pred_fields, x_grid, axis=1)
        r2_cl_int = r2_score(y_test_scalars[:, 0], cl_integrated)
        print(f"Physics Consistency: Integrated Cp Lift vs True CL -> R² = {r2_cl_int:.4f}")

        # Save model weights
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mlp_model.state_dict(), out_dir / "best_surrogate_scalars.pt")
        torch.save(field_model.state_dict(), out_dir / "best_surrogate_cp.pt")
        print(f"\nTrained PyTorch model checkpoints saved to '{out_dir}/'")

    except ImportError:
        print("\n[Notice] PyTorch not installed in current environment.")
        print("This script is ready to run on Kaggle / GPU environments with PyTorch!")

    # -------------------------------------------------------------
    # Model 3: Scikit-Learn Gradient Boosting Baseline
    # -------------------------------------------------------------
    from sklearn.ensemble import GradientBoostingRegressor

    print("\nTraining Model 3 (Gradient Boosted Trees Baseline)...")
    for i, target in enumerate(target_cols):
        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        gbr.fit(X[idx_train], y_scalars[idx_train, i])
        pred_gbr = gbr.predict(X[idx_test])
        r2_gbr = r2_score(y_scalars[idx_test, i], pred_gbr)
        rmse_gbr = np.sqrt(mean_squared_error(y_scalars[idx_test, i], pred_gbr))
        print(f"  GBR {target.upper():<4} -> R² = {r2_gbr:.4f} | RMSE = {rmse_gbr:.5e}")

    print("\n" + "=" * 72)
    print("SURROGATE MODEL TRAINING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
