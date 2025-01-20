import polars as pl

def target_data_engineering(path):
    df = pl.read_csv(path)
    df = df.with_columns(
        df["ID"]
        .str.extract(r'\d+', 0)
        .cast(pl.Int64)
        .alias("ID")
    )
    df = df.sort("ID")
    df = df.drop("ID")
    return df

def clinical_data_engineering(path):
    """
    1) sort the data by ID column
    2) drop CENTER column (because only one center in the test dataset)
    3) normalize numerical data
    4) extract features from CYTOGENETICS column
    """
    df = pl.read_csv(path)
    df = df.with_columns(df["ID"].str.extract(r'\d+', 0)
                        .cast(pl.Int64).alias("ID")
        )
    df = df.sort("ID")
    df = df.drop("ID")
    df = df.drop("CENTER")
    df = df.with_columns(df["MONOCYTES"].cast(pl.Float64))
    numerical_columns = [c for c in df.columns if df[c].dtype == pl.Float64]
    for c in numerical_columns:
        if c == "HB":
            pass
        else:
            df = df.with_columns(
                ((1 + df[c]).log()).alias(f"log({c})")
                )
    for c in df.columns:
        if df[c].dtype == pl.Float64:
            df = df.with_columns(
                ((df[c] - df[c].mean()) / df[c].std()).alias(f"Normalized({c})")
                )
    for c in numerical_columns:
        df = df.drop(c)

    for c in df.columns:
        if "log" in c:
            df = df.drop(c)

    df = df.with_columns([
            df[c].fill_null(0.0).alias(c)
            for c in df.columns]
        )

    df = df.with_columns([
        (pl.col("CYTOGENETICS").str.contains(r'\b46\b')).cast(pl.Int64).alias("Normal_Count"),
        (pl.col("CYTOGENETICS").str.contains(r'Normal')).cast(pl.Int64).alias("Normal"),
        
        (pl.col("CYTOGENETICS").str.contains(r'\bXX\b')).cast(pl.Int64).alias("Female"),
        (pl.col("CYTOGENETICS").str.contains(r'\bXY\b')).cast(pl.Int64).alias("Male"),
        (pl.col("CYTOGENETICS").str.contains(r'\bX,-Y\b')).cast(pl.Int64).alias("Lost_Y"),
        (pl.col("CYTOGENETICS").str.contains(r'\bX,-X\b')).cast(pl.Int64).alias("Lost_X"),
        
        (pl.col("CYTOGENETICS").str.contains(r't')).cast(pl.Int64).alias("Translocations"),
        (pl.col("CYTOGENETICS").str.contains(r'add')).cast(pl.Int64).alias("Additional"),
        (pl.col("CYTOGENETICS").str.contains(r'der')).cast(pl.Int64).alias("Derivative"),
        (pl.col("CYTOGENETICS").str.contains(r'mar')).cast(pl.Int64).alias("Marker"),
        (pl.col("CYTOGENETICS").str.contains(r'del')).cast(pl.Int64).alias("Deletion"),
    ])
    df = df.drop("CYTOGENETICS")
    return df

def molecular_data_engineering(path, traindata=True):
    """
    1) sort the data by ID column
    """
    if traindata:
        df = pl.read_csv(path)
    else:
        df = pl.read_csv(path, schema_overrides={"CHR": pl.Utf8})
    df = df.with_columns(
        df["ID"]
        .str.extract(r'\d+', 0)
        .cast(pl.Int64)
        .alias("ID")
    )
    df = df.sort("ID")
    df = df.drop("ID")
    return df

SAVE = True

if SAVE:
    Modified_Trainclinical = clinical_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Trainclinical.csv")
    Modified_Trainmolecular = molecular_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Trainmolecular.csv")
    Modified_Ytrain = target_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Ytrain.csv")

    Modified_Testclinical = clinical_data_engineering("Desktop/QubeChallenge/RawData/TestDataset/Testclinical.csv")
    Modified_Testmolecular = molecular_data_engineering("Desktop/QubeChallenge/RawData/TestDataset/Testmolecular.csv", traindata=False)

    Modified_Trainclinical.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Trainclinical.csv")
    Modified_Trainmolecular.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Trainmolecular.csv")
    Modified_Ytrain.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

    Modified_Testclinical.write_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testclinical.csv")
    Modified_Testmolecular.write_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testmolecular.csv")