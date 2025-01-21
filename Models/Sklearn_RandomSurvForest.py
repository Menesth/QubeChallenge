import numpy as np
import polars as pl
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw

np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtrain = Xtraindf.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)
kf = KFold(n_splits=10, shuffle=True, random_state=1337)

n_estimators_list = [100]
max_depth_list = [10]
min_samples_split_list = [6]
min_samples_leaf_list = [3]
max_features_list = ["sqrt"]

for max_features in max_features_list:
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                for min_samples_leaf in min_samples_leaf_list:
                    fold_concordance_indices = []

                    for train_idx, val_idx in kf.split(Xtrain):
                        Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
                        ytr, yval = ytrain[train_idx], ytrain[val_idx]

                        model = RandomSurvivalForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            random_state=1337,
                        )
                        model.fit(Xtr, ytr)
                        concordance_index = concordance_index_ipcw(ytr, yval, model.predict(Xval))[0]
                        fold_concordance_indices.append(concordance_index)
                    mean_concordance = np.mean(fold_concordance_indices)
                    std_concordance = np.std(fold_concordance_indices)
                    print(
                        f"n_estimators = {n_estimators}, max_depth = {max_depth},"
                        f"min_samples_split = {min_samples_split}, min_samples_leaf = {min_samples_leaf},"
                        f"max_features = {max_features}"
                    )
                    print(f"Mean Concordance Index = {mean_concordance:.4f}, Std = {std_concordance:.4f}\n")

submit = False
if submit:
    Xtestdf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testdataset.csv")
    Xtest = Xtestdf.to_numpy()
    ytestpred = model.predict(Xtest)

    id_column = [f"KYW{i+1}" for i in range(len(ytestpred))]
    submission = pl.DataFrame({
        "ID": id_column,
        "risk_score": ytestpred
    })
    submission.write_csv("Desktop/QubeChallenge/RSFsubmission.csv")