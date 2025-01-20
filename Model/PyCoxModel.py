import torch
import numpy as np
import polars as pl
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.model_selection import train_test_split

torch.manual_seed(1337)

featuresdf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
targetdf = pl.read_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

get_target = lambda df: (df['OS_YEARS'].to_numpy().astype(np.float32), df['OS_STATUS'].to_numpy())

Xtrain = featuresdf.to_numpy().astype(np.float32)
ytrain = get_target(targetdf)

Xtr, Xval, ytr, yval = train_test_split(Xtrain, ytrain, test_size=0.3, random_state=1337)
val_data = Xval, yval

in_features = Xtrain.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
batch_size = 16
dropout = 0.1
output_bias = False
lr = 1e-3

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
model = CoxPH(net, tt.optim.Adam)

model.optimizer.set_lr(lr)
log = model.fit(Xtr, ytr, batch_size, epochs=1000,
                val_data=val_data, val_batch_size=batch_size)