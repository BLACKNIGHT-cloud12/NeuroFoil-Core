# LinkedIn post — NeuroFoil

## Variant A (main post)

I built an end-to-end aerodynamic ML platform — from raw OpenFOAM simulations
to a desktop app with honest uncertainty quantification.

What it does:
• Runs the automated 504-point RANS campaign design (21 airfoils × 4
  Reynolds numbers × 6 angles of attack) with checkpointed persistence and
  quality gates — 400 cases converged, the rest excluded on physical evidence
• Trains neural surrogates for force coefficients AND full surface-pressure
  fields (161-point Cp distributions)
• Serves predictions through a local desktop app — the 3D wing is painted by
  the model's own pressure field and morphs with the geometry sliders

The part I care most about — honesty:
• Grouped per-airfoil holdouts (airfoils never seen in training):
  R² 0.966 (CL) / 0.933 (CD) / 0.860 (CM)
• Split-conformal intervals: empirical coverage 0.951 / 0.918 / 0.967 / 0.957
  against the 0.95 nominal level
• A Mahalanobis domain guard flags out-of-domain queries instead of
  pretending — and 104 operating points were excluded after I verified from
  force histories that they are steady-RANS limit cycles, not bugs

Also shipped: a mesh-sensitivity study, an automated QC audit, physics
consistency checks (integrated ΔCp lift vs CL: R² 0.983), and a 12-page
technical report.

Everything is open: github.com/BLACKNIGHT-cloud12/NeuroFoil-Core

#MachineLearning #CFD #OpenFOAM #ComputationalFluidDynamics #Aerospace
#ScientificML #DeepLearning

---

## Variant B (short)

A 504-point campaign design → 400 converged RANS cases → neural surrogates
→ a desktop "flight deck" where the wing is painted by its own predicted
pressure field.

Uncertainty is measured, not claimed: conformal coverage 0.951/0.918/0.967
(CL/CD/CM) on airfoils the model never saw. 104 unsteady operating points
excluded on evidence, not convenience.

Full pipeline + technical report:
github.com/BLACKNIGHT-cloud12/NeuroFoil-Core

#CFD #MachineLearning #OpenFOAM #Aerospace

---

## Posting notes
- Attach linkedin/neurofoil_linkedin_banner.png (1200×627, LinkedIn's native
  link-image ratio) as the post image.
- Best times: Tue–Thu morning in your target timezone.
- If asked "is this validated?" — link the technical report
  (report/NeuroFoil_Technical_Report.pdf in the repo); every number in the
  post is traceable to it.
- Don't post the ± bands as "error bars prove accuracy" — they are calibrated
  intervals; the wording above stays within that claim.
