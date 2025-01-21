import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtraindf = Xtraindf.drop("ID")
ytraindf = ytraindf.drop("ID")

Xtrain = Xtraindf.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

kf = KFold(n_splits=10, shuffle=True, random_state=1337)

alpha_list = [1e-3]
ties_list = ["efron"]
n_iter_list = [100]

for alpha in alpha_list:
    for ties in ties_list:
        for n_iter in n_iter_list:
            fold_concordance_indices = []

            for train_idx, val_idx in kf.split(Xtrain):
                Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
                ytr, yval = ytrain[train_idx], ytrain[val_idx]

                model = CoxPHSurvivalAnalysis(
                    alpha=alpha,
                    ties=ties,
                    n_iter=n_iter
                )

                model.fit(Xtr, ytr)
                concordance_index = concordance_index_ipcw(ytr, yval, model.predict(Xval))[0]
                fold_concordance_indices.append(concordance_index)

            mean_concordance = np.mean(fold_concordance_indices)
            std_concordance = np.std(fold_concordance_indices)
            print(f"alpha = {alpha}, ties = {ties}, n_iter = {n_iter}")
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
    submission.write_csv("Desktop/QubeChallenge/submission.csv")