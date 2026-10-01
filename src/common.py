"""Shared loaders and fixed-hyper-parameter models for the supplementary experiments."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run from repo root regardless of cwd
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

SEED = 42

def load_enb():
    d = pd.read_csv("data/ENB2012_data.csv")
    X = d[[f"X{i}" for i in range(1, 9)]].values
    g = pd.factorize(d[["X1", "X2", "X3", "X4", "X5"]].astype(str).agg("|".join, axis=1))[0]
    return X, {"Y1": d.Y1.values, "Y2": d.Y2.values}, g

def load_concrete():
    d = pd.read_csv("data/concrete_data.csv"); d.columns = [c.strip() for c in d.columns]
    for c in d.columns: d[c] = pd.to_numeric(d[c].astype(str).str.strip())
    X = d.iloc[:, :8].values
    g = pd.factorize(d.iloc[:, :7].astype(str).agg("|".join, axis=1))[0]
    return X, {"strength": d.iloc[:, 8].values}, g

def load_pk(features="all"):
    d = pd.read_csv("data/parkinsons_updrs.data")
    voice = [c for c in d.columns if c not in ("subject#", "age", "sex", "test_time", "total_UPDRS", "motor_UPDRS")]
    sets = {"all": ["age", "sex", "test_time"] + voice, "voice": voice, "voice+demo": ["age", "sex"] + voice,
            "voice+time": ["test_time"] + voice, "demo+time": ["age", "sex", "test_time"]}
    X = d[sets[features]].values
    return X, {"total_UPDRS": d.total_UPDRS.values, "motor_UPDRS": d.motor_UPDRS.values}, d["subject#"].values, d

def fixed_model(name):
    if name == "Ridge": return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "SVR": return make_pipeline(StandardScaler(), SVR(C=30.0))
    if name == "kNN": return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9))
    if name == "RF": return RandomForestRegressor(n_estimators=100, max_features=0.5, random_state=SEED, n_jobs=1)
    if name == "GBM": return GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED)
    raise KeyError(name)

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
