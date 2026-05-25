from functools import lru_cache
from pathlib import Path

import pandas as pd


DATASET_PATH = Path("telecomunicatii.csv")


@lru_cache(maxsize=1)
def get_telecom_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def get_dataset_summary() -> dict:
    if not DATASET_PATH.exists():
        return {
            "status": "missing",
            "path": str(DATASET_PATH),
        }

    data = get_telecom_dataset()
    target_counts = data["payment_delay"].value_counts().to_dict()

    return {
        "status": "available",
        "path": str(DATASET_PATH),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "target": "payment_delay",
        "target_distribution": {
            str(label): int(count)
            for label, count in target_counts.items()
        },
        "features": [
            column
            for column in data.columns
            if column != "payment_delay"
        ],
    }
