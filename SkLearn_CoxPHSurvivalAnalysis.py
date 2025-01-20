import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split

np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtrain = Xtraindf.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

Xtr, Xval, ytr, yval = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=1337)

model = CoxPHSurvivalAnalysis(ties="efron", alpha=1e-3, n_iter=100)
model.fit(Xtr, ytr)

cox_cindex_train = concordance_index_ipcw(ytr, ytr, model.predict(Xtr), tau=2)[0]
cox_cindex_val = concordance_index_ipcw(ytr, yval, model.predict(Xval), tau=2)[0]

print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on val: {cox_cindex_val:.2f}")

Xtestdf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testdataset.csv")
Xtest = Xtestdf.to_numpy()
ytestpred = model.predict(Xtest)

id_column = [f"KYW{i+1}" for i in range(len(ytestpred))]
submission = pl.DataFrame({
    "ID": id_column,
    "risk_score": ytestpred
})
submission.write_csv("Desktop/QubeChallenge/submission.csv")