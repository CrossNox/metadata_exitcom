import shutil
from tqdm import tqdm_notebook as tqdm
import gzip
from youconfigme import AutoConfig
import pandas as pd
from pathlib import Path
import os
import requests


# taken from youconfigme's cast_utils
def ensure_path(path):
    """Create a path if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest, override=False, chunksize=4096):
    if os.path.exists(dest) and not override:
        return
    with requests.get(url, stream=True) as r:
        try:
            file_size = int(r.headers["Content-Length"])
        except KeyError:
            file_size = 0
        chunks = file_size // chunksize

        with open(dest, "wb") as f, tqdm(
            total=file_size, unit="iB", unit_scale=True
        ) as t:
            for chunkdata in r.iter_content(chunksize):
                f.write(chunkdata)
                t.update(len(chunkdata))


config = AutoConfig()
DATA_FOLDER = config.metadata_exitcom.data_folder(cast=ensure_path)
PREDICTIONS_FOLDER = config.metadata_exitcom.predictions_folder(cast=ensure_path)

DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?id={gid}&export=download".format
GSPREADHSEET_DOWNLOAD_URL = (
    "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
)

TRAIN_CSV = DATA_FOLDER / "train.csv"
VALIDATION_CSV = DATA_FOLDER / "validate.csv"

TRAIN_URL = DRIVE_DOWNLOAD_URL(gid=config.metadata_exitcom.train_gid())
VALIDATION_URL = DRIVE_DOWNLOAD_URL(gid=config.metadata_exitcom.test_gid())

EXAMPLE_URL = DRIVE_DOWNLOAD_URL(gid=config.metadata_exitcom.example_gid())
EXAMPLE_CSV = PREDICTIONS_FOLDER / "example.csv"


def init_data():
    # download data

    data = {TRAIN_CSV: TRAIN_URL, VALIDATION_CSV: VALIDATION_URL}

    for item_path, url in data.items():
        download_file(url, item_path)

    download_file(EXAMPLE_URL, EXAMPLE_CSV)
