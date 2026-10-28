"""
OpenFOAM Case Builder for 2D Airfoil RANS Simulations (Milestone 1.3).

Generates complete 0/, constant/, and system/ OpenFOAM dictionaries tailored
for steady incompressible turbulent aerodynamic flows (simpleFoam, k-omega SST).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatingPoint:
    """Operating conditions for 2D aerodynamic simulation."""

    alpha_deg: float
    reynolds: float
    u_inf: float = 1.0
    chord_m: float = 1.0
    rho: float = 1.0

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.alpha_deg)

    @property
    def nu(self) -> float:
        """Kinematic viscosity nu = (U_inf * c) / Re."""
        return (self.u_inf * self.chord_m) / self.reynolds

    @property
    def velocity_vector(self) -> tuple[float, float, float]:
        """Cartesian velocity components (Ux, Uy, Uz)."""
        ux = self.u_inf * math.cos(self.alpha_rad)
        uy = self.u_inf * math.sin(self.alpha_rad)
        return (ux, uy, 0.0)

    @property
    def lift_dir(self) -> tuple[float, float, float]:
        """Lift direction unit vector perpendicular to freestream."""
        return (-math.sin(self.alpha_rad), math.cos(self.alpha_rad), 0.0)

    @property
    def drag_dir(self) -> tuple[float, float, float]:
        """Drag direction unit vector parallel to freestream."""
        return (math.cos(self.alpha_rad), math.sin(self.alpha_rad), 0.0)

    def turbulence_inlet_values(
        self,
        intensity: float = 0.001,
        eddy_viscosity_ratio: float = 0.1,
    ) -> tuple[float, float, float]:
        """
        Calculate freestream turbulent quantities: (k, omega, nut).
        - k_inf = 1.5 * (U_inf * intensity)^2
        - nut_inf = eddy_viscosity_ratio * nu
        - omega_inf = k_inf / nut_inf
        """
        k_inf = 1.5 * (self.u_inf * intensity) ** 2
        nut_inf = eddy_viscosity_ratio * self.nu
        omega_inf = k_inf / max(nut_inf, 1e-15)
        return k_inf, omega_inf, nut_inf


def build_openfoam_case(
    case_dir: Path | str,
    op: OperatingPoint,
    cfg: dict[str, Any],
) -> None:
    """
    Construct the full OpenFOAM case directory structure (0/, constant/, system/)
    configured for simpleFoam with the specified OperatingPoint and config dictionary.
    """
    case_path = Path(case_dir)
    zero_dir = case_path / "0"
    constant_dir = case_path / "constant"
    system_dir = case_path / "system"

    for d in (zero_dir, constant_dir, system_dir):
        d.mkdir(parents=True, exist_ok=True)

    op_cfg = cfg.get("operating_conditions", {})
    solver_cfg = cfg.get("solver", {})
    mesh_cfg = cfg.get("mesh", {})

    intensity = float(op_cfg.get("turbulence_intensity", 0.001))
    ev_ratio = float(op_cfg.get("eddy_viscosity_ratio", 0.1))
    k_inf, omega_inf, nut_inf = op.turbulence_inlet_values(intensity, ev_ratio)

    ux, uy, _ = op.velocity_vector
    lx, ly, _ = op.lift_dir
    dx, dy, _ = op.drag_dir

    span_thickness = float(mesh_cfg.get("span_thickness_m", 0.1))
    a_ref = op.chord_m * span_thickness

    turb_model = solver_cfg.get("turbulence_model", "kOmegaSST")
    max_iter = int(solver_cfg.get("max_iterations", 1500))
    write_interval = int(solver_cfg.get("write_interval", 1500))
    purge_write = int(solver_cfg.get("purge_write", 1))

    res_cfg = solver_cfg.get("residual_targets", {})
    res_p = float(res_cfg.get("p", 1e-5))
    res_u = float(res_cfg.get("U", 1e-6))
    res_k = float(res_cfg.get("k", 1e-6))
    res_w = float(res_cfg.get("omega", 1e-6))

    # 1. 0/U
    u_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({ux:.8e} {uy:.8e} 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           $internalField;
    }}
    airfoil
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    (zero_dir / "U").write_text(u_content, encoding="utf-8", newline="\n")

    # 2. 0/p
    p_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    airfoil
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""
    (zero_dir / "p").write_text(p_content, encoding="utf-8", newline="\n")

    # 3. 0/k
    k_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inf:.8e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    airfoil
    {{
        type            kqRWallFunction;
        value           uniform 1e-10;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    (zero_dir / "k").write_text(k_content, encoding="utf-8", newline="\n")

    # 4. 0/omega
    omega_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inf:.8e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    airfoil
    {{
        type            omegaWallFunction;
        blended         true;
        value           $internalField;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    (zero_dir / "omega").write_text(omega_content, encoding="utf-8", newline="\n")

    # 5. 0/nut
    nut_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut_inf:.8e};

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           $internalField;
    }}
    outlet
    {{
        type            calculated;
        value           $internalField;
    }}
    airfoil
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    (zero_dir / "nut").write_text(nut_content, encoding="utf-8", newline="\n")

    # 6. constant/transportProperties
    tp_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {op.nu:.8e};
"""
    (constant_dir / "transportProperties").write_text(tp_content, encoding="utf-8", newline="\n")

    # 7. constant/turbulenceProperties
    turb_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType  RAS;

RAS
{{
    RASModel        {turb_model};
    turbulence      on;
    printCoeffs     on;
}}
"""
    (constant_dir / "turbulenceProperties").write_text(turb_content, encoding="utf-8", newline="\n")

    # 8. system/controlDict with forceCoeffs function object
    control_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {max_iter};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      {purge_write};
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             yes;
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          {op.rho:.4f};
        liftDir         ({lx:.8f} {ly:.8f} 0);
        dragDir         ({dx:.8f} {dy:.8f} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {op.u_inf:.4f};
        lRef            {op.chord_m:.4f};
        Aref            {a_ref:.6f};
    }}
}}
"""
    (system_dir / "controlDict").write_text(control_content, encoding="utf-8", newline="\n")

    # 9. system/fvSchemes
    schemes_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         corrected;
}
wallDist
{
    method          meshWave;
}
"""
    (system_dir / "fvSchemes").write_text(schemes_content, encoding="utf-8", newline="\n")

    # 10. system/fvSolution
    solution_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
    }}
    Phi
    {{
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
    }}
    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.01;
    }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 1;
    residualControl
    {{
        p               {res_p:.1e};
        U               {res_u:.1e};
        k               {res_k:.1e};
        omega           {res_w:.1e};
    }}
}}
relaxationFactors
{{
    fields
    {{
        p               0.1;
    }}
    equations
    {{
        U               0.3;
        k               0.5;
        omega           0.5;
    }}
}}
"""
    (system_dir / "fvSolution").write_text(solution_content, encoding="utf-8", newline="\n")
