import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42

DOTA_ZIP = DATA_DIR / "dota2+games+results.zip"
CHESS_ZIP = DATA_DIR / "chess+king+rook+vs+king+pawn.zip"

DOTA_META = ["cluster_id", "game_mode", "game_type"]
DOTA_HEROES = [f"hero_{i}" for i in range(1, 114)]
DOTA_COLUMNS = ["team1_win"] + DOTA_META + DOTA_HEROES

CHESS_FEATURES = [
    "bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp",
    "bxqsq", "cntxt", "dsopp", "dwipd", "hdchk", "katri", "mulch", "qxmsq", "r2ar8",
    "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl", "skach", "skewr", "skrxp",
    "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg",
]


def _read_zip_csv(path, member, **kwargs):
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        return pd.read_csv(fh, **kwargs)


def load_dota_raw():
    train = _read_zip_csv(DOTA_ZIP, "dota2Train.csv", header=None, names=DOTA_COLUMNS)
    test = _read_zip_csv(DOTA_ZIP, "dota2Test.csv", header=None, names=DOTA_COLUMNS)
    return train, test


def load_dota():
    train, test = load_dota_raw()
    y_train = (train.pop("team1_win") == 1).astype(int).to_numpy()
    y_test = (test.pop("team1_win") == 1).astype(int).to_numpy()
    return train, test, y_train, y_test


def load_chess_raw():
    return _read_zip_csv(CHESS_ZIP, "kr-vs-kp.data", header=None, names=CHESS_FEATURES + ["outcome"])


def load_chess(test_size=0.2):
    df = load_chess_raw()
    y = (df.pop("outcome") == "won").astype(float).to_numpy()
    return train_test_split(df, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
