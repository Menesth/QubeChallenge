import numpy as np
import polars as pl
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw

np.random.seed(1337)

Xtrain_df = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtrain_df = Xtrain_df.drop("ID")
ytraindf = ytraindf.drop("ID")

Xtrain = Xtrain_df.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

kf = KFold(n_splits=5, shuffle=True, random_state=1337)

fold_concordance_indices_tr = []
fold_concordance_indices_val = []
for train_idx, val_idx in kf.split(Xtrain):
    Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
    ytr, yval = ytrain[train_idx], ytrain[val_idx]

    model = RandomSurvivalForest(
        n_estimators=200,
        max_depth=15,
        min_samples_split = 6,
        min_samples_leaf = 3,
        min_weight_fraction_leaf = 0,
        max_features = 0.8,
        random_state = 1337,
        max_samples = 0.8
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
print(f"Train mean = {mean_concordance_tr:.4f}, Train std = {std_concordance_tr:.4f}\n")
print(f"Val mean = {mean_concordance_val:.4f}, Val std = {std_concordance_val:.4f}\n")