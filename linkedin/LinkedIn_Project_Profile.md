# NeuroFoil — LinkedIn Projects section copy

## 1. Project name

NeuroFoil — Real-CFD Surrogate Platform with Conformal Uncertainty

Alternatives:
- NeuroFoil Flight Deck — OpenFOAM Campaign to Deployed ML
- NeuroFoil: Aerodynamic Surrogates Trained on 400 Real RANS Cases

## 2. Description (Project description field)

Short (~240 chars):
End-to-end aerodynamic ML platform: a 504-point OpenFOAM RANS campaign
design, 400 converged cases, neural surrogates for force coefficients and
full surface-pressure fields, split-conformal intervals measured on unseen
airfoils, and a local desktop flight deck.

Long (~600 chars, for About or a pinned comment):
NeuroFoil is an end-to-end aerodynamic data engine. It runs an automated
504-point OpenFOAM RANS campaign (21 NACA airfoils × 4 Reynolds numbers ×
6 angles of attack, checkpointed per case) — 400 cases converged; 104
operating points were excluded after verification showed steady-RANS limit
cycles. Surrogates predict CL/CD/CM and 161-point Cp fields. Evaluation
uses grouped per-airfoil holdouts: R² 0.966/0.933/0.860 on unseen
geometries, physics-consistency R² 0.983 between integrated ΔCp and CL,
and split-conformal coverage of 0.951/0.918/0.967/0.957 against the 0.95
nominal level. Ships with a FastAPI+Electron desktop application, a
12-page technical report, and a QC/mesh-sensitivity toolchain. Every
number is traceable to a repository artifact.

## 3. Skills to attach (in order)

  1. Computational Fluid Dynamics (CFD)
  2. OpenFOAM
  3. Machine Learning
  4. Deep Learning
  5. PyTorch
  6. Uncertainty Quantification
  7. Conformal Prediction
  8. Scientific Machine Learning (SciML)
  9. Aerodynamics
 10. Reynolds-Averaged Navier–Stokes (RANS)
 11. Python
 12. FastAPI
 13. Electron.js
 14. Data Visualization
 15. Software Development

## 4. Featured-section one-liner

"NeuroFoil — 400 converged OpenFOAM cases; neural surrogates with measured
conformal coverage (0.951/0.918/0.967) on unseen airfoils. Open source."

## 5. Scope statement (use verbatim if asked)

Research tool. 2D incompressible RANS, NACA four-digit family,
Re 1e6–6e6. Not airworthiness or design certification. No temperature
results (incompressible isothermal campaign); a compressible
energy-equation campaign is planned.
