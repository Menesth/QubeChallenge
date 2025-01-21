import numpy as np
import polars as pl
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtrain = Xtraindf.to_numpy()
ytrain = np.array(
    [(bool(event), time) for event, time in zip(ytraindf["OS_STATUS"], ytraindf["OS_YEARS"])],
    dtype=[("event", "bool"), ("time", "float64")]
)

Xtr, Xval, ytr, yval = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=1337)

for n_estimator in [50, 100, 150]:
    for max_depth in [5, 10, None]:
        for min_samples_split in [6]:
            for min_samples_leaf in [3]:

                model = RandomSurvivalForest(n_estimators=n_estimator, max_depth=max_depth, min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf, max_features='sqrt', max_leaf_nodes=None, random_state=1337)
                model.fit(Xtr, ytr)
                concordance_index = concordance_index_ipcw(ytr, yval, model.predict(Xval))
                print("n_estimator =", n_estimator, "max_depth =", max_depth, "min_samples_split =", min_samples_split, "min_samples_leaf =", min_samples_leaf)
                print("Val concordance_index =", concordance_index[0])

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