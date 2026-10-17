#!/usr/bin/env python3
"""
Generate and Package Aerodynamic CFD Dataset for Kaggle Training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.dataset.pipeline import generate_physics_augmented_dataset


def main() -> None:
    print("=" * 72)
    print("GENERATING AERODYNAMIC DATASET FOR KAGGLE SURROGATE TRAINING")
    print("=" * 72)

    csv_path, npz_path = generate_physics_augmented_dataset(
        n_samples=500,
        seed=42,
    )

    df = pd.read_csv(csv_path)
    npz = np.load(npz_path)

    print(f"Generated CSV dataset : {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"Generated NPZ dataset : {npz_path} ({npz_path.stat().st_size / 1024:.1f} KB)")
    print(f"Total samples         : {len(df)}")
    print(f"Feature columns ({len(df.columns)}) : {list(df.columns[:8])} ... {list(df.columns[-6:])}")
    print("\nTarget Statistics:")
    print(df[["cl", "cd", "cm", "lift_to_drag"]].describe().round(4))
    print("\nSurface Cp Arrays:")
    print(f"  x_grid shape        : {npz['x_grid'].shape}")
    print(f"  cp_upper shape      : {npz['cp_upper'].shape}")
    print(f"  cp_lower shape      : {npz['cp_lower'].shape}")
    print(f"  delta_cp shape      : {npz['delta_cp'].shape}")
    print("=" * 72)
    print("DATASET READY FOR KAGGLE UPLOAD")
    print("=" * 72)


if __name__ == "__main__":
    main()
