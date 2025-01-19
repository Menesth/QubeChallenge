import polars as pl

Raw_Trainclinical = "/Users/thibautmenes/Desktop/QubeChallenge/RawData/TrainDataset/Trainclinical.csv"
Raw_Trainmolecular = "/Users/thibautmenes/Desktop/QubeChallenge/RawData/TrainDataset/Trainmolecular.csv"
Raw_Ytrain = "/Users/thibautmenes/Desktop/QubeChallenge/RawData/TrainDataset/Ytrain.csv"

df_trainclinical = pl.read_csv(Raw_Trainclinical)
df_trainmolecular = pl.read_csv(Raw_Trainmolecular)
dfYtrain = pl.read_csv(Raw_Ytrain)

df_trainclinical = df_trainclinical.sort("ID")
df_trainmolecular = df_trainmolecular.sort("ID")
dfYtrain = dfYtrain.sort("ID")

df_trainclinical.write_csv("/Users/thibautmenes/Desktop/QubeChallenge/ModifiedData/TrainDataset/Trainclinical.csv")
df_trainmolecular.write_csv("/Users/thibautmenes/Desktop/QubeChallenge/ModifiedData/TrainDataset/Trainmolecular.csv")
dfYtrain.write_csv("/Users/thibautmenes/Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

Raw_Testclinical = "/Users/thibautmenes/Desktop/QubeChallenge/RawData/TestDataset/Testclinical.csv"
Raw_Testmolecular = "/Users/thibautmenes/Desktop/QubeChallenge/RawData/TestDataset/Testmolecular.csv"

df_testclinical = pl.read_csv(Raw_Testclinical)
df_testmolecular = pl.read_csv(Raw_Testmolecular, schema_overrides={"CHR": pl.Utf8})

df_testclinical = df_testclinical.sort("ID")
df_testmolecular = df_testmolecular.sort("ID")

df_testclinical.write_csv("/Users/thibautmenes/Desktop/QubeChallenge/ModifiedData/TestDataset/Testclinical.csv")
df_testmolecular.write_csv("/Users/thibautmenes/Desktop/QubeChallenge/ModifiedData/TestDataset/Testmolecular.csv")