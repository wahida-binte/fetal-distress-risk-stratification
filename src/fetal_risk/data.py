from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path, target: str = "fetal_health", drop_duplicates: bool = False):
    """Load the CTG dataset and return (X, y) with a 0-indexed target.

    Labels: 0=Normal, 1=Suspect, 2=Pathological.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[target])
    y = (df[target] - 1).astype(int)
    return X, y
