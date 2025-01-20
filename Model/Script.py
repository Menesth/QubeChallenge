import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split

np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Trainclinical.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

combined = Xtraindf.with_columns([ytraindf["OS_STATUS"], ytraindf["OS_YEARS"]])
filtered = combined.filter(~pl.col("OS_YEARS").is_null())

Xtraindf = filtered.select([col for col in filtered.columns if col not in ["OS_YEARS", "OS_STATUS"]])
ytraindf = filtered.select([col for col in filtered.columns if col in ["OS_YEARS", "OS_STATUS"]])

Xtrain = Xtraindf.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

Xtr, Xval, ytr, yval = train_test_split(Xtrain, ytrain, test_size=0.3, random_state=1337)
Xval, Xte, yval, yte = train_test_split(Xval, yval, test_size=0.5, random_state=1337)

model = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200)
model.fit(Xtr, ytr)

cox_cindex_train = concordance_index_ipcw(ytr, ytr, model.predict(Xtr), tau=7)[0]
cox_cindex_val = concordance_index_ipcw(ytr, yval, model.predict(Xval), tau=7)[0]

print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on val: {cox_cindex_val:.2f}")