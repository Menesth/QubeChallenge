import numpy as np
import torch
import polars as pl
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import KFold
from pycox.models import CoxPH
import torchtuples as tt

torch.manual_seed(1337)
np.random.seed(1337)

Xtraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
ytraindf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Xtraindf = Xtraindf.drop("ID")
ytraindf = ytraindf.drop("ID")

Xtrain = Xtraindf.to_numpy()
ytrain = ytraindf.to_numpy()

Xtrain = Xtrain.astype(np.float32)
ytrain = ytrain.astype(np.float32)

kf = KFold(n_splits=10, shuffle=True, random_state=1337)

in_features = Xtrain.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                            batch_norm=batch_norm, dropout=dropout, output_bias=output_bias)

epochs = 500
batch_size = 64
callbacks = [tt.callbacks.EarlyStopping()]
learning_rate_list = [1e-4]

#np.array([(bool(event), time) for event, time in zip(ytr[:, 0], ytr[:, 1])],dtype=[("event", "bool"), ("time", "float32")])

for learning_rate in learning_rate_list:
    fold_concordance_indices = []
    for train_idx, val_idx in kf.split(Xtrain):
        Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
        ytr, yval = ytrain[train_idx], ytrain[val_idx]
        ytr = ytr[:, 0], ytr[:, 1]
        yval = np.array([(bool(event), time) for event, time in zip(yval[:, 0], yval[:, 1])],dtype=[("event", "bool"), ("time", "float32")])
        val_data = Xval, yval

        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(learning_rate)
        model.fit(Xtr, ytr, batch_size, epochs, callbacks, verbose=False, val_data=val_data, val_batch_size=batch_size)
        
        concordance_index = concordance_index_ipcw(ytr, yval, model.predict(Xval))[0]
        fold_concordance_indices.append(concordance_index)
        print(concordance_index)
        exit()

    mean_concordance = np.mean(fold_concordance_indices)
    std_concordance = np.std(fold_concordance_indices)
    print(f"learning_rate = {learning_rate}")
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