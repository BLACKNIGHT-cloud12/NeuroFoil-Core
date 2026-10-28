# Running the NeuroFoil CFD Campaign on Another Machine

This package contains everything needed to generate the real OpenFOAM CFD
dataset. The pipeline runs on **Windows + WSL2** or **native Linux**, and is
**resume-safe**: if it is interrupted, re-run the same command and it skips
every case already completed.

---

## 1. What you need

| Requirement | Details |
|---|---|
| OS | Windows 10/11 with WSL2, **or** native Linux |
| OpenFOAM | **v2606** (ESI / openfoam.com build) |
| Python | 3.10 – 3.12 with `numpy`, `scipy`, `pandas`, `pyyaml` |
| Disk | ~2 GB free (cases are trimmed as they finish) |
| RAM | 8 GB+ (each solver instance is small; 3–5 can run in parallel) |

> OpenFOAM version note: if you install a different version (e.g. v2506),
> edit `configs/default_cfd.yaml` → `wsl_environment.openfoam_bashrc` and set
> the correct path (e.g. `/usr/lib/openfoam/openfoam2506/etc/bashrc`).

---

## 2. Environment setup

### 2a. Windows machine

```powershell
# In an Administrator PowerShell:
wsl --install -d Ubuntu-24.04
# ...reboot if prompted, then open the Ubuntu terminal:
```

Inside the **Ubuntu (WSL) terminal**:

```bash
# OpenFOAM v2606 (ESI build — same as the development machine)
curl -s https://dl.openfoam.com/install/add-openfoam-repo.sh | sudo bash
sudo apt update
sudo apt install -y openfoam2606-default
# Verify:
source /usr/lib/openfoam/openfoam2606/etc/bashrc && which simpleFoam blockMesh
```

On the **Windows side**: install Python from python.org (tick "Add to PATH"),
then:

```powershell
pip install numpy scipy pandas pyyaml
```

### 2b. Linux machine

```bash
curl -s https://dl.openfoam.com/install/add-openfoam-repo.sh | sudo bash
sudo apt update && sudo apt install -y openfoam2606-default python3-pip
pip install numpy scipy pandas pyyaml
```

> On native Linux the pipeline calls `bash` directly (no WSL needed) — the
> code detects the OS automatically.

---

## 3. Run the campaign

1. Unzip this package somewhere **outside OneDrive/Dropbox** (sync churn with
   hundreds of case folders hurts). Example: `C:\NeuroFoil` or `~/neurofoil`.
2. From the package root:

```bash
# Windows (PowerShell) or Linux — identical command
python scripts/run_dataset_campaign.py --airfoils 21 --alphas="-6,-2,2,6,10,14" --res="1e6,2e6,3e6,6e6" --jobs 5
```

- `--jobs 5` = five solver instances in parallel. Use 3 on a 4-core machine,
  up to half your logical cores otherwise.
- 504 cases total, roughly **8–14 h** depending on CPU.
- You can **Ctrl+C at any time** and re-run the same command to resume.
- Each finished case prints one line: case id, convergence status, CL, CD.
  Final statuses (`converged` / `unconverged`) are preserved per row — that is
  intentional and honest; do not delete "unconverged" rows.

Optional smoke test first (4 cases, ~15 min):

```bash
python scripts/run_dataset_campaign.py --limit 4 --jobs 2
```

Expected for the first NACA 0012 cases at Re 6e6: α=0° → CL ≈ 0.00, CD ≈ 0.0088;
α=4° → CL ≈ 0.439, CD ≈ 0.0095. If you see wildly different values (CL ±10),
stop and check the OpenFOAM install (`bash tests/../../scripts/_pitz_ref.sh`
equivalent: run any `simpleFoam` tutorial to confirm the solver works).

---

## 4. What to send back

Copy these two files back to the main machine (into the repo's
`data/processed/` folder, replacing the synthetic placeholders):

```
data/processed/airfoil_cfd_dataset.csv      # 504 rows, honest per-case status
data/processed/cp_surface_distributions.npz # Cp fields for converged cases
```

Optionally also keep `data/raw_cases/` (solver logs + postProcessing, ~100 MB)
as raw evidence for the paper's data-availability statement.

If you also want the grid-convergence study for the paper, run on the same
machine afterwards:

```bash
python scripts/run_single_case.py          # NACA0012 medium, prints summary
```

(The coarse/medium/fine study was already run on the development machine;
numbers are recorded in the repo README once updated.)

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `wsl` not found / distro missing | `wsl --install -d Ubuntu-24.04`, or edit `configs/default_cfd.yaml` → `wsl_environment.distro` |
| `simpleFoam: command not found` | wrong `openfoam_bashrc` path in `configs/default_cfd.yaml` |
| Cases all fail in <5 s | run `wsl` once manually and confirm OpenFOAM sources; check `data/raw_cases/<case>/log.blockMesh` |
| Extremely slow | reduce `--jobs` if RAM-bound; keep the folder off cloud-synced paths |
| Interrupted | just re-run the same command — completed cases are skipped |
