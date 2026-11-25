import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pytest

from fuzzyrisk.data import Dataset


def make_synthetic(n=240, seed=0) -> Dataset:
    """Small mixed-type clinical-like dataset (numeric with NaN, binary, nominal)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 12, n)
    bmi = rng.normal(28, 5, n)
    glu = rng.normal(110, 30, n)
    sex = rng.choice(["male", "female"], n)
    cp = rng.choice(["a", "b", "c"], n)
    z = -0.5 + 0.06 * (age - 55) + 0.08 * (bmi - 28) + 0.03 * (glu - 110) + 0.5 * (sex == "male") \
        + 0.8 * (cp == "c")
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    X = pd.DataFrame({"age": age, "bmi": bmi, "glu": glu, "sex": sex, "cp": cp})
    X.loc[rng.choice(n, 12, replace=False), "bmi"] = np.nan
    X.loc[rng.choice(n, 5, replace=False), "cp"] = np.nan
    X["sex"] = X["sex"].astype(object)
    X["cp"] = X["cp"].astype(object)
    return Dataset(name="synthetic", X=X, y=y, numeric_cols=["age", "bmi", "glu"],
                   binary_cols=["sex"], nominal_cols=["cp"],
                   categories={"sex": ["female", "male"], "cp": ["a", "b", "c"]})


@pytest.fixture(scope="session")
def synth():
    return make_synthetic()


@pytest.fixture(scope="session")
def cols(synth):
    return dict(numeric_cols=synth.numeric_cols, binary_cols=synth.binary_cols,
                nominal_cols=synth.nominal_cols, categories=synth.categories)


FAST_CFG = {"pop_size": 12, "n_gen": 3, "n_iter": 2, "R_max": 6, "L_max": 3}
