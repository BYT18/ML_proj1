"""Data loading and feature engineering for the BRFSS MICHD prediction task.

The pipeline is fitted on the training set only (category lists, medians,
quantile bin edges, standardization statistics) and then applied unchanged to
validation / test data.
"""

import os

import numpy as np

from helpers import load_csv_data

# Survey bookkeeping columns (dates, sequence numbers, sampling weights, phone
# screening). They describe how the interview was conducted, not the person.
ADMIN_COLUMNS = {
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "CTELNUM1",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_LLCPWT",
    "_DUALCOR",
    "_DUALUSE",
}

# BRFSS answer codes for "don't know / not sure" and "refused" in raw
# (non-calculated) numeric questions, and the code for "none".
UNKNOWN_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}
NONE_CODES = {88, 888}


def load_data(data_path="data"):
    """Load the competition CSVs, caching them as a .npz for faster reloads.

    Returns:
        x_train, x_test, y_train (in {-1, 1}), train_ids, test_ids, feature names.
    """
    cache = os.path.join(data_path, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        arrays = d["xtr"], d["xte"], d["ytr"], d["tri"], d["tei"]
    else:
        arrays = load_csv_data(data_path)
        np.savez(
            cache,
            xtr=arrays[0],
            xte=arrays[1],
            ytr=arrays[2],
            tri=arrays[3],
            tei=arrays[4],
        )
    with open(os.path.join(data_path, "x_train.csv")) as f:
        names = f.readline().strip().split(",")[1:]
    return (*arrays, names)


def clean_numeric(col, name):
    """Map BRFSS special codes of a raw numeric question to NaN / 0.

    Calculated variables (names starting with '_') are already clean. For raw
    questions, a code is treated as special only if it lies above the bulk of
    the answers (e.g. 77/99 for a 1-30 day count, 7777/9999 for weight).
    """
    col = col.copy()
    if name.startswith("_"):
        return col
    valid = col[~np.isnan(col)]
    if valid.size == 0:
        return col
    # Largest value that is not a special code: special codes sit above it.
    regular = valid[~np.isin(valid, list(UNKNOWN_CODES | NONE_CODES))]
    top = regular.max() if regular.size else 0
    for code in NONE_CODES:
        if code > top:
            col[col == code] = 0
    for code in UNKNOWN_CODES:
        if code > top:
            col[col == code] = np.nan
    return col


class FeatureBuilder:
    """Turns the raw 321 BRFSS columns into a numeric design matrix.

    Options (used for the ablation study):
        drop_admin:   drop survey bookkeeping columns.
        one_hot:      one-hot encode low-cardinality (categorical) columns;
                      otherwise they are treated like numeric columns.
        bins:         number of quantile bins for numeric columns (0 = none).
        max_nan:      drop columns whose missing rate exceeds this value.
        max_cat:      columns with at most this many distinct values are
                      considered categorical.
        min_freq:     categories rarer than this fraction are not encoded.
    """

    def __init__(
        self,
        drop_admin=True,
        one_hot=True,
        bins=10,
        max_nan=1.0,
        max_cat=30,
        min_freq=1e-3,
    ):
        self.drop_admin = drop_admin
        self.one_hot = one_hot
        self.bins = bins
        self.max_nan = max_nan
        self.max_cat = max_cat
        self.min_freq = min_freq

    def fit(self, x, names):
        """Learn the encoding (column types, categories, bins, statistics)."""
        n = x.shape[0]
        self.specs = []  # one entry per kept raw column
        for j, name in enumerate(names):
            col = x[:, j]
            nan_rate = np.isnan(col).mean()
            if self.drop_admin and name in ADMIN_COLUMNS:
                continue
            if nan_rate > self.max_nan or nan_rate == 1.0:
                continue
            values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
            if len(values) <= 1 and nan_rate == 0:
                continue  # constant column carries no information
            if self.one_hot and len(values) <= self.max_cat:
                cats = values[counts >= self.min_freq * n]
                self.specs.append(
                    {"j": j, "name": name, "type": "cat", "cats": cats, "nan": nan_rate}
                )
            else:
                c = clean_numeric(col, name)
                valid = c[~np.isnan(c)]
                spec = {
                    "j": j,
                    "name": name,
                    "type": "num",
                    "median": np.median(valid),
                    "nan": np.isnan(c).mean(),
                }
                # Clip extreme values at the 0.5/99.5 percentiles before scaling.
                spec["lo"], spec["hi"] = np.percentile(valid, [0.5, 99.5])
                clipped = np.clip(valid, spec["lo"], spec["hi"])
                spec["mean"], spec["std"] = clipped.mean(), clipped.std() + 1e-12
                if self.bins:
                    qs = np.linspace(0, 100, self.bins + 1)[1:-1]
                    spec["edges"] = np.unique(np.percentile(valid, qs))
                self.specs.append(spec)
        return self

    def transform(self, x, dtype=np.float32):
        """Build the design matrix (without the bias column)."""
        blocks, self.feature_names = [], []
        for s in self.specs:
            col = x[:, s["j"]]
            name = s["name"]
            if s["type"] == "cat":
                blocks.append(col[:, None] == s["cats"][None, :])
                self.feature_names += [f"{name}={v:g}" for v in s["cats"]]
                if s["nan"] > 0:
                    blocks.append(np.isnan(col)[:, None])
                    self.feature_names.append(f"{name}=NaN")
                continue
            c = clean_numeric(col, name)
            missing = np.isnan(c)
            filled = np.where(missing, s["median"], c)
            z = (np.clip(filled, s["lo"], s["hi"]) - s["mean"]) / s["std"]
            blocks.append(z[:, None])
            self.feature_names.append(name)
            if s["nan"] > 0:
                blocks.append(missing[:, None])
                self.feature_names.append(f"{name}=NaN")
            if self.bins and len(s["edges"]):
                # Bin index in 0..len(edges); missing values get no bin.
                idx = np.searchsorted(s["edges"], filled, side="right")
                onehot = idx[:, None] == np.arange(len(s["edges"]) + 1)[None, :]
                onehot[missing] = False
                blocks.append(onehot)
                self.feature_names += [
                    f"{name}_bin{k}" for k in range(len(s["edges"]) + 1)
                ]
        return np.hstack([b.astype(dtype) for b in blocks])


def add_bias(x):
    """Prepend a column of ones."""
    return np.hstack([np.ones((x.shape[0], 1), dtype=x.dtype), x])
