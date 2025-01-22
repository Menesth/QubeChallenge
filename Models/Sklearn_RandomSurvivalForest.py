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

kf = KFold(n_splits=10, shuffle=True, random_state=1337)

fold_concordance_indices = []
for train_idx, val_idx in kf.split(Xtrain):
    Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
    ytr, yval = ytrain[train_idx], ytrain[val_idx]

    model = RandomSurvivalForest(
        n_estimators=100,
        max_depth=15,
        min_samples_split = 6,
        min_samples_leaf = 3,
        min_weight_fraction_leaf = 0,
        max_features = "sqrt",
        max_leaf_nodes = None,
        random_state = 1337,
        max_samples = None
    )

    model.fit(Xtr, ytr)
    predicted_risk = model.predict(Xval)
    concordance_index = concordance_index_ipcw(ytr, yval, predicted_risk, tau=7)[0]
    fold_concordance_indices.append(concordance_index)

mean_concordance = np.mean(fold_concordance_indices)
std_concordance = np.std(fold_concordance_indices)
print(f"Mean = {mean_concordance:.4f}, Std = {std_concordance:.4f}\n")