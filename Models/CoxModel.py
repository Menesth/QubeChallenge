import sys
import os
import numpy as np
import polars as pl
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw

sys.path.append(os.path.abspath("Desktop/QubeChallenge/Preprocessing"))
from Script import preprocessing, normalize # type: ignore

Xtrain_df = preprocessing("Desktop/QubeChallenge/RawData/TrainDataset/Trainclinical.csv", "Desktop/QubeChallenge/RawData/TrainDataset/Trainmolecular.csv")
Xtrain_df = normalize(Xtrain_df)
ytraindf = pl.read_csv("Desktop/QubeChallenge/RawData/TrainDataset/Ytrain.csv")

# cleaning: ytraindf has few empty rows
ids_to_keep = ytraindf.filter(~pl.col("OS_YEARS").is_null() & ~pl.col("OS_STATUS").is_null())["ID"]
Xtrain_df = Xtrain_df.filter(pl.col("ID").is_in(ids_to_keep))
ytraindf = ytraindf.filter(pl.col("ID").is_in(ids_to_keep))

# drop ID column
Xtrain_df = Xtrain_df.drop("ID")
ytraindf = ytraindf.drop("ID")

# get numpy arrays
Xtrain = Xtrain_df.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

# k-fold cross validation
np.random.seed(1337)
kf = KFold(n_splits=10, shuffle=True, random_state=1337)

fold_concordance_indices_tr = []
fold_concordance_indices_val = []
for train_idx, val_idx in kf.split(Xtrain):
    Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
    ytr, yval = ytrain[train_idx], ytrain[val_idx]

    model = CoxPHSurvivalAnalysis(
        alpha = 1e-5,
        ties = "breslow",
        n_iter=100
    )

    model.fit(Xtr, ytr)
    predicted_risk_tr = model.predict(Xtr)
    predicted_risk_val = model.predict(Xval)

    concordance_index_tr = concordance_index_ipcw(ytr, ytr, predicted_risk_tr, tau=7)[0]
    concordance_index_val = concordance_index_ipcw(ytr, yval, predicted_risk_val, tau=7)[0]
    fold_concordance_indices_tr.append(concordance_index_tr)
    fold_concordance_indices_val.append(concordance_index_val)

mean_concordance_tr = np.mean(fold_concordance_indices_tr)
std_concordance_tr = np.std(fold_concordance_indices_tr)
mean_concordance_val = np.mean(fold_concordance_indices_val)
std_concordance_val = np.std(fold_concordance_indices_val)
print(f"train mean = {mean_concordance_tr:.4f}, train std = {std_concordance_tr:.4f}\n")
print(f"val mean = {mean_concordance_val:.4f}, val std = {std_concordance_val:.4f}\n")