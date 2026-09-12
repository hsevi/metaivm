"""Select 40 diverse datasets from the registry for manageable clustering runs."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

registry = pd.read_csv(DATA_DIR / "dataset_registry.csv")

# Only keep preprocessed datasets
processed = [
    d for d in registry["dataset_id"]
    if (DATA_DIR / "processed" / d / "X.npy").exists()
]
registry = registry[registry["dataset_id"].isin(processed)].copy()

# Strategy: pick all synthetic (29) + diverse subset of OpenML (11)
# For OpenML: pick diverse n_classes, n_samples, n_features
synthetic = registry[registry["source"] == "synthetic"]
openml = registry[registry["source"] == "openml"].copy()

# Sort OpenML by diversity — pick a spread across n_classes and n_samples
# Hand-pick to cover: binary, multi-class, many-class; small, medium, large; low-d, high-d
openml_picks = [
    "openml_15",    # breast-w: n=699, d=9, k=2 (small, low-d, binary)
    "openml_37",    # diabetes: n=768, d=8, k=2
    "openml_1510",  # wdbc: n=569, d=30, k=2 (small, moderate-d)
    "openml_54",    # vehicle: n=846, d=18, k=4
    "openml_307",   # vowel: n=990, d=12, k=11 (many classes)
    "openml_181",   # yeast: n=1484, d=8, k=10 (many classes)
    "openml_1462",  # banknote: n=1372, d=4, k=2 (low-d)
    "openml_40984", # segment: n=2310, d=16, k=7
    "openml_182",   # satimage: n=6430, d=36, k=6 (medium, moderate-d)
    "openml_28",    # optdigits: n=5620, d=64, k=10 (high-d, many classes)
    "openml_40966", # MiceProtein: n=1080, d=77, k=8 (high-d)
    "openml_458",   # authorship: n=841, d=70, k=4 (high-d)
    "openml_40499", # texture: n=5500, d=40, k=11
    "openml_1497",  # wall-robot: n=5456, d=24, k=4
    "openml_1489",  # phoneme: n=5404, d=5, k=2 (medium, low-d)
]

selected_openml = openml[openml["dataset_id"].isin(openml_picks)]

selected = pd.concat([synthetic, selected_openml], ignore_index=True)
print(f"Selected {len(selected)} datasets: {len(synthetic)} synthetic + {len(selected_openml)} OpenML")
print(f"\nn_samples range: {selected['n_samples'].min()} - {selected['n_samples'].max()}")
print(f"n_features range: {selected['n_features'].min()} - {selected['n_features'].max()}")
print(f"n_classes range: {selected['n_classes'].min()} - {selected['n_classes'].max()}")

# Save selected subset
selected.to_csv(DATA_DIR / "dataset_registry_selected.csv", index=False)
print(f"\nSaved to {DATA_DIR / 'dataset_registry_selected.csv'}")
print("\nDatasets:")
for _, row in selected.iterrows():
    print(f"  {row['dataset_id']}: n={row['n_samples']}, d={row['n_features']}, k={row['n_classes']}")
