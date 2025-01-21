import numpy as np
import polars as pl
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest, ExtraSurvivalTrees, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw

np.random.seed(1337)

Xtrain_MINMAX_df = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset_MINMAXNORM.csv")
Xtrain_STD_df = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset_STDNORM.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtrain_MINMAX_df = Xtrain_MINMAX_df.drop("ID")
Xtrain_STD_df = Xtrain_STD_df.drop("ID")
ytraindf = ytraindf.drop("ID")

Xtrain_MINMAX = Xtrain_MINMAX_df.to_numpy()
Xtrain_STD = Xtrain_STD_df.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

Xtr_MINMAX, Xval_MINMAX, ytr_MINMAX, yval_MINMAX = train_test_split(Xtrain_MINMAX, ytrain, test_size=0.25, random_state=1337)
Xtr_STD, Xval_STD, ytr_STD, yval_STD = train_test_split(Xtrain_STD, ytrain, test_size=0.25, random_state=1337)

GBSAmodel = GradientBoostingSurvivalAnalysis(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=3,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                dropout_rate=0,
                                max_features="log2",
                                random_state=1337
                                )
CWGBSAmodel = ComponentwiseGradientBoostingSurvivalAnalysis(
                                n_estimators=100,
                                learning_rate=0.1,
                                dropout_rate=0,
                                subsample=1,
                                random_state=1337
                                )
RFSAmodel = RandomSurvivalForest(
                                n_estimators=100,
                                max_depth=None,
                                min_samples_split=6,
                                min_samples_leaf=3,
                                max_samples=None,
                                max_features="log2",
                                n_jobs=-1,
                                random_state=1337
                                )
ERFSAmodel = ExtraSurvivalTrees(
                                n_estimators=100,
                                max_depth=None,
                                min_samples_split=6,
                                min_samples_leaf=3,
                                max_samples=None,
                                max_features="log2",
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
CWGBSAmodel.fit(Xtr, ytr)
RFSAmodel.fit(Xtr, ytr)
ERFSAmodel.fit(Xtr, ytr)
CoxPHmodel.fit(Xtr, ytr)
Coxnetmodel.fit(Xtr, ytr)

GBSApred = GBSAmodel.predict(Xval)
CWGBSApred = CWGBSAmodel.predict(Xval)
RFSApred = RFSAmodel.predict(Xval)
ERFSApred = ERFSAmodel.predict(Xval)
CovPHpred = CoxPHmodel.predict(Xval)
Covnetpred = Coxnetmodel.predict(Xval)

concordance_index_GBSA = concordance_index_ipcw(ytr, yval, GBSApred)[0]
concordance_index_CWGBSA = concordance_index_ipcw(ytr, yval, CWGBSApred)[0]
concordance_index_RFSA = concordance_index_ipcw(ytr, yval, RFSApred)[0]
concordance_index_ERFSA = concordance_index_ipcw(ytr, yval, ERFSApred)[0]
concordance_index_CoxPH = concordance_index_ipcw(ytr, yval, CovPHpred)[0]
concordance_index_Coxnet = concordance_index_ipcw(ytr, yval, Covnetpred)[0]
print(f"concordance_index_GBSA = {concordance_index_GBSA:.4f}, concordance_index_CWGBSA = {concordance_index_CWGBSA:.4f}, concordance_index_RFSA = {concordance_index_RFSA:.4f}, concordance_index_RFSA = {concordance_index_ERFSA:.4f}")
print(f"concordance_index_CoxPH = {concordance_index_CoxPH:.4f}, concordance_index_Coxnet = {concordance_index_Coxnet:.4f}")

GBSApred = (GBSApred - GBSApred.min()) / (GBSApred.max() - GBSApred.min())
CWGBSApred = (CWGBSApred - CWGBSApred.min()) / (CWGBSApred.max() - CWGBSApred.min())
RFSApred = (RFSApred - RFSApred.min()) / (RFSApred.max() - RFSApred.min())
ERFSApred = (ERFSApred - ERFSApred.min()) / (ERFSApred.max() - ERFSApred.min())
CovPHpred = (CovPHpred - CovPHpred.min()) / (CovPHpred.max() - CovPHpred.min())
Covnetpred = (Covnetpred - Covnetpred.min()) / (Covnetpred.max() - Covnetpred.min())

concordance_index_sum = concordance_index_GBSA + concordance_index_CWGBSA + concordance_index_RFSA + concordance_index_ERFSA + concordance_index_CoxPH + concordance_index_Coxnet
pred = (concordance_index_GBSA * GBSApred) + (concordance_index_CWGBSA * CWGBSApred) + (concordance_index_RFSA * RFSApred) + (concordance_index_ERFSA * ERFSApred) + (concordance_index_CoxPH * CovPHpred) + (concordance_index_Coxnet * Covnetpred)
pred = pred / concordance_index_sum

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