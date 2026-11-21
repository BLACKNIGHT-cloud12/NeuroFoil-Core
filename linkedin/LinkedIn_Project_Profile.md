# NeuroFoil — LinkedIn Projects section copy

## 1. Project name (pick one)

A. NeuroFoil — Real-CFD Surrogate Platform with Conformal Uncertainty   ← recommended
B. NeuroFoil Flight Deck — From OpenFOAM Campaign to Certified-Honest ML
C. NeuroFoil: Aerodynamic Surrogates Trained on 400 Real RANS Cases

(LinkedIn "Project name" field — use A. Keep "NeuroFoil" as the searchable
token; the suffix carries the differentiators.)

## 2. Description (Project description field)

Short (~240 chars):
End-to-end aerodynamic ML platform: a 504-point OpenFOAM RANS campaign
design, 400 converged cases, neural surrogates for force coefficients and
full surface-pressure fields, split-conformal intervals measured on unseen
airfoils, and a local desktop flight deck.

Long (~600 chars, for About or a pinned comment):
NeuroFoil is an end-to-end aerodynamic data engine. It plans and runs an
automated 504-point OpenFOAM RANS campaign (21 NACA airfoils × 4 Reynolds
numbers × 6 angles of attack, checkpointed per case), curates the result
into a provenance-controlled dataset (400/504 covered; 104 unsteady
operating points excluded on evidence, not convenience), and trains
surrogates for CL/CD/CM and full 161-point Cp fields. Evaluation uses
grouped per-airfoil holdouts — R² 0.966/0.933/0.860 on unseen geometries,
physics-consistency R² 0.983 between integrated ΔCp and CL — with
split-conformal intervals measured at 0.951/0.918/0.967/0.957 against the
0.95 nominal level. Everything ships: a FastAPI+Electron desktop flight
deck where the wing is painted by live predictions, a 12-page technical
report, and a QC/mesh-sensitivity toolchain. Every number is traceable to
a repository artifact.

## 3. Skills to attach (ranked; add to post + your profile)

Post tags (LinkedIn allows a few per post):
  Computational Fluid Dynamics · OpenFOAM · Machine Learning ·
  Scientific Machine Learning · Aerospace

Profile skills for this project (in order):
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

## 4. Featured-section one-liner (if you pin it)

"NeuroFoil — 400 converged OpenFOAM cases → neural surrogates with measured
0.95-nominal conformal coverage on unseen airfoils. Open source."

## 5. Do / don't
- DO keep "400 converged / 504-point design / 104 excluded" — the honesty is
  the differentiator and technical readers check it.
- DON'T write "certified", "validated for design use", or "production-ready
  CFD" — the README and report deliberately scope this as a research tool.
- DON'T claim aerodynamic heating results — the thermal note in the app
  explains why they don't exist yet (compressible v2 campaign).
