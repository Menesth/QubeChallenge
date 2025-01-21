import numpy as np
import polars as pl
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw

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

Xtr, Xval, ytr, yval = train_test_split(Xtrain, ytrain, test_size=0.25, random_state=1337)

GBSAmodel = GradientBoostingSurvivalAnalysis(
                                            n_estimators=100,
                                            learning_rate=0.1,
                                            max_depth=3,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            dropout_rate=0,
                                            max_features="sqrt",
                                            random_state=1337
                                            )
RFSAmodel = RandomSurvivalForest(
                                n_estimators=100,
                                max_depth=None,
                                min_samples_split=6,
                                min_samples_leaf=3,
                                max_samples=None,
                                max_features="sqrt",
                                n_jobs=-1,
                                random_state=1337
                                )
CoxPHmodel = CoxPHSurvivalAnalysis(
                                    alpha=1e-3,
                                    ties="efron",
                                    n_iter=100
                                )
Coxnetmodel = CoxnetSurvivalAnalysis(
                                n_alphas=100,
                                alphas=None,
                                l1_ratio=0.5,
                                penalty_factor=None,
                                max_iter=100000
                            )

GBSAmodel.fit(Xtr, ytr)
RFSAmodel.fit(Xtr, ytr)
CoxPHmodel.fit(Xtr, ytr)
Coxnetmodel.fit(Xtr, ytr)

GBSApred = GBSAmodel.predict(Xval)
RFSApred = RFSAmodel.predict(Xval)
CovPHpred = CoxPHmodel.predict(Xval)
Covnetpred = Coxnetmodel.predict(Xval)

concordance_index_GBSA = concordance_index_ipcw(ytr, yval, GBSApred)[0]
concordance_index_RFSA = concordance_index_ipcw(ytr, yval, RFSApred)[0]
concordance_index_CoxPH = concordance_index_ipcw(ytr, yval, CovPHpred)[0]
concordance_index_Coxnet = concordance_index_ipcw(ytr, yval, Covnetpred)[0]

print(f"concordance_index_GBSA = {concordance_index_GBSA:.4f}, concordance_index_RFSA = {concordance_index_RFSA:.4f}, concordance_index_CoxPH = {concordance_index_CoxPH:.4f}, concordance_index_Coxnet = {concordance_index_Coxnet:.4f}")

GBSApred = (GBSApred - GBSApred.min()) / (GBSApred.max() - GBSApred.min())
RFSApred = (RFSApred - RFSApred.min()) / (RFSApred.max() - RFSApred.min())
CovPHpred = (CovPHpred - CovPHpred.min()) / (CovPHpred.max() - CovPHpred.min())
Covnetpred = (Covnetpred - Covnetpred.min()) / (Covnetpred.max() - Covnetpred.min())
pred = GBSApred + RFSApred + CovPHpred + Covnetpred
concordance_index_sum = concordance_index_ipcw(ytr, yval, pred)[0]
print(f"concordance_index_sum = {concordance_index_sum:.4f}")

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