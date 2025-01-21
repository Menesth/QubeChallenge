import numpy as np
import polars as pl
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import KFold
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

kf = KFold(n_splits=5, shuffle=True, random_state=1337)

n_estimators_list = [200]
max_depth_list = [5]
max_features_list = ["log2"]

dropout_rate_list = [0]
min_samples_split_list = [2]
min_samples_leaf_list = [1]

ccp_alpha_list = [0]

learning_rate_list = [1e-1]

normalization_list = ["STD", "MINMAX"]

for normalization in normalization_list:
    for max_features in max_features_list:
        for learning_rate in learning_rate_list:
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    for dropout_rate in dropout_rate_list:
                        for min_samples_split in min_samples_split_list:
                            for min_samples_leaf in min_samples_leaf_list:
                                for ccp_alpha in ccp_alpha_list:
                                    fold_concordance_indices = []
                                    if normalization == "STD":
                                        Xtrain = Xtrain_STD
                                    else:
                                        Xtrain = Xtrain_MINMAX
                                    for train_idx, val_idx in kf.split(Xtrain):
                                        Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
                                        ytr, yval = ytrain[train_idx], ytrain[val_idx]

                                        model = GradientBoostingSurvivalAnalysis(
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            dropout_rate=dropout_rate,
                                            max_features=max_features,
                                            random_state=1337
                                        )
                                        model.fit(Xtr, ytr)
                                        concordance_index = concordance_index_ipcw(ytr, yval, model.predict(Xval))[0]
                                        fold_concordance_indices.append(concordance_index)
                                    mean_concordance = np.mean(fold_concordance_indices)
                                    std_concordance = np.std(fold_concordance_indices)
                                    print(
                                        f"n_estimators = {n_estimators}, max_depth = {max_depth}, learning_rate = {learning_rate}, "
                                        f"min_samples_split = {min_samples_split}, min_samples_leaf = {min_samples_leaf}, dropout_rate = {dropout_rate}, "
                                        f"max_features = {max_features}, ccp_alpha = {ccp_alpha}"
                                    )
                                    print(f"normalization = {normalization}")
                                    print(f"Mean = {mean_concordance:.4f}, Std = {std_concordance:.4f}\n")

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
    submission.write_csv("Desktop/QubeChallenge/GDSAsubmission.csv")