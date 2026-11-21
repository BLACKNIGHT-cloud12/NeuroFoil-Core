# LinkedIn post — NeuroFoil

## Variant A (main post)

I built an end-to-end aerodynamic ML platform — from automated OpenFOAM
simulations to a local desktop application.

What it does:
• Runs a 504-point RANS campaign (21 NACA airfoils × 4 Reynolds numbers ×
  6 angles of attack) with per-case checkpointing and quality gates.
  400 cases converged; 104 were excluded after verification showed
  steady-RANS limit cycles.
• Trains neural surrogates for force coefficients and 161-point
  surface-pressure fields.
• Serves predictions in a desktop app: a 3D wing colored by the live
  pressure field, conformal intervals, and a Mahalanobis domain guard.

Measured results — grouped per-airfoil holdouts (airfoils excluded from
training):
• R² 0.966 (CL) / 0.933 (CD) / 0.860 (CM)
• Surface-pressure R² 0.892
• Conformal coverage 0.951 / 0.918 / 0.967 / 0.957 vs 0.95 nominal
• Integrated ΔCp lift vs CL: R² 0.983

Also included: mesh-sensitivity study, automated QC audit, 12-page
technical report.

Code and report: github.com/BLACKNIGHT-cloud12/NeuroFoil-Core

#ComputationalFluidDynamics #OpenFOAM #MachineLearning #SciML #Aerospace

---

## Variant B (short)

504-point RANS campaign design. 400 converged cases. Neural surrogates for
force coefficients and surface-pressure fields.

Grouped-holdout R²: CL 0.966, CD 0.933, CM 0.860.
Conformal coverage 0.951 / 0.918 / 0.967 / 0.957 vs 0.95 nominal.

Code and technical report:
github.com/BLACKNIGHT-cloud12/NeuroFoil-Core

#CFD #OpenFOAM #MachineLearning #Aerospace

---

## Posting notes
- Attach linkedin/neurofoil_linkedin_banner.png (1200×627).
- Best times: Tue–Thu morning in your target timezone.
- Coverage figures are conformal interval coverage, not accuracy claims.
