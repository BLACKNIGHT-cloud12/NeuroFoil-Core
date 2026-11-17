from pathlib import Path
import ast
import json

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_kaggle_notebook_v2.py"
NOTEBOOK = REPO / "notebooks" / "neurofoil_surrogate_training_v2.ipynb"


def _load_builder_and_notebook() -> tuple[str, dict, str, list[str]]:
    builder = BUILDER.read_text(encoding="utf-8")
    ast.parse(builder, filename=str(BUILDER))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    return builder, notebook, "\n\n".join(code_cells), code_cells


def test_training_notebook_field_scaler_contract() -> None:
    builder, _notebook, code, _code_cells = _load_builder_and_notebook()

    required_contract = (
        "Xtr_field = scaler_x_g.transform(X[mask_tr_cp])",
        "Xva_field = scaler_x_g.transform(X[mask_va_cp])",
        "Xte_field = scaler_x_g.transform(X[mask_te_cp])",
        "Xcal_field = scaler_x_g.transform(X[mask_va_cp])",
        "field_model, Xtr_field, ytr_field, Xva_field, yva_field",
        "field_model(torch.tensor(Xte_field)",
        "field_model(torch.tensor(Xcal_field)",
        "np.save(out_dir / 'scaler_field_mean.npy', field_g.mean_)",
        "np.save(out_dir / 'scaler_field_scale.npy', field_g.scale_)",
        "'has_field_scaler': True",
        "'field_input_scaler': 'scaler_x'",
        "'field_target_scaler': 'scaler_field'",
    )
    for snippet in required_contract:
        assert snippet in builder, f"missing builder contract: {snippet}"
        assert snippet in code, f"missing notebook contract: {snippet}"

    forbidden_raw_paths = (
        "train_model(field_model, X[mask_tr_cp]",
        "field_model(torch.tensor(X[mask_te_cp])",
        "field_model(torch.tensor(X[is_val & has_cp])",
    )
    for snippet in forbidden_raw_paths:
        assert snippet not in builder
        assert snippet not in code


def test_training_notebook_validates_inputs_before_training() -> None:
    builder, _notebook, code, code_cells = _load_builder_and_notebook()

    validation_contract = (
        "FEATURE_ORDER + TARGET_ORDER + ['status', 'naca_code', 'case_id', 'data_source']",
        "CSV contains no converged OpenFOAM rows for training.",
        'All converged training rows must have data_source="openfoam"',
        "if 'is_synthetic' in df_raw.columns:",
        "refusing to train the canonical OpenFOAM model",
        "if len(unique_codes) < 6:",
        "mask_tr_cp.any() and mask_va_cp.any() and mask_te_cp.any()",
        "expected_field_shape = (npz_case_ids_raw.size, x_grid.size)",
        "Cp NPZ case_ids must be unique.",
        "CSV case_id values must be unique",
        "Cp NPZ contains case_ids absent from the CSV",
    )
    for snippet in validation_contract:
        assert snippet in builder, f"missing builder validation: {snippet}"
        assert snippet in code, f"missing notebook validation: {snippet}"

    first_training_cell = next(
        index for index, source in enumerate(code_cells) if "train_model(mlp_grouped" in source
    )
    for snippet in validation_contract:
        assert any(snippet in source for source in code_cells[:first_training_cell])

    provenance_metadata = (
        "'data_source': 'openfoam'",
        "'is_synthetic': False",
        "'source_row_count': int(len(df))",
        "'source_status_counts':",
    )
    for snippet in provenance_metadata:
        assert snippet in builder, f"missing builder provenance metadata: {snippet}"
        assert snippet in code, f"missing notebook provenance metadata: {snippet}"


def test_training_notebook_ood_audit_uses_standardized_features() -> None:
    builder, _notebook, code, _code_cells = _load_builder_and_notebook()

    normalized_ood_contract = (
        "feat_tr_norm = scaler_x_g.transform(X[is_train])",
        "cov_norm = np.cov(feat_tr_norm, rowvar=False)",
        "Xq_norm = scaler_x_g.transform(Xq)",
        "d = Xq_norm - mu_norm",
        "mahal_normalized(X[is_train])",
        "mahal_normalized(X[is_test])",
    )
    for snippet in normalized_ood_contract:
        assert snippet in builder, f"missing normalized builder OOD audit: {snippet}"
        assert snippet in code, f"missing normalized notebook OOD audit: {snippet}"

    forbidden_raw_covariance = (
        "feat_tr = X[is_train]",
        "cov = np.cov(feat_tr, rowvar=False)",
        "d = Xq - mu",
    )
    for snippet in forbidden_raw_covariance:
        assert snippet not in builder
        assert snippet not in code
