import polars as pl

def target_data_engineering(path):
    df = pl.read_csv(path)
    df = df.with_columns(
            df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID")
        )
    df = df.sort("ID")
    return df

def clinical_data_engineering(path):

    df = pl.read_csv(path)
    df = df.with_columns(
            df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID")
        )
    df = df.sort("ID")
    df = df.drop(["CENTER"])
    df = df.with_columns(df["MONOCYTES"].cast(pl.Float64))

    for c in df.columns:
        if df[c].dtype == pl.Float64:
            if c == "HB":
                pass
            else:
                    df = df.with_columns(
                            ((1 + df[c]).log()).alias(c)
                        )
    for c in df.columns:
        if df[c].dtype == pl.Float64:
            df = df.with_columns(
                        ((df[c] - df[c].min()) / (df[c].max() - df[c].min())).alias(c)
                    )
            df = df.with_columns(
                        df[c].fill_null(df[c].median()).alias(c)
                    )

    df = df.with_columns([
        (pl.col("CYTOGENETICS").str.contains(r'\b46\b')).cast(pl.Int64).alias("Normal_Chromo_Count"),
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
    for c in df.columns:
        if df[c].dtype == pl.Int64:
            df = df.with_columns(
                        df[c].fill_null(0).alias(c)
                    )
    df = df.drop("CYTOGENETICS")
    return df

def molecular_data_engineering(path, traindata=True):

    if traindata:
        df = pl.read_csv(path)
        RANGE = 132729 + 1
    else:
        df = pl.read_csv(path, schema_overrides={"CHR": pl.Utf8})
        RANGE = 1193 + 1
    df = df.with_columns(
            df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID")
        )
    df = df.sort("ID")

    df = df.with_columns(
            df["VAF"].fill_null(df["VAF"].median()).alias("VAF")
        )
    df = df.with_columns(
            df["DEPTH"].fill_null(df["DEPTH"].median()).alias("DEPTH")
        )
    df = df.with_columns(
            (df['VAF'] * df["DEPTH"]).alias("VAF_*_DEPTH")
        )
    df = df.with_columns(
            ((df["VAF_*_DEPTH"] - df["VAF_*_DEPTH"].min()) / (df["VAF_*_DEPTH"].max() - df["VAF_*_DEPTH"].min())).alias("VAF_*_DEPTH")
        )
    df = df.with_columns(
            ((df["DEPTH"] - df["DEPTH"].min()) / (df["DEPTH"].max() - df["DEPTH"].min())).alias("DEPTH")
        )
    df = df.with_columns(
            (df['END'] - df["START"] + 1).alias("MUTATION_LEN")
        )
    df = df.with_columns(
            df["MUTATION_LEN"].fill_null(0.0).alias("MUTATION_LEN")
        )
    df = df.with_columns([
        (pl.col("EFFECT").str.contains(r'ITD')).cast(pl.Int64).alias("ITD"),
        (pl.col("EFFECT").str.contains(r'PTD')).cast(pl.Int64).alias("PTD"),
        
        (pl.col("EFFECT").str.contains(r'codon_loss')).cast(pl.Int64).alias("CODON_LOSS"),
        (pl.col("EFFECT").str.contains(r'codon_gain')).cast(pl.Int64).alias("CODON_GAIN"),
        (pl.col("EFFECT").str.contains(r'synonymous_codon')).cast(pl.Int64).alias("CODON_SYNONYMOUS"),

        (pl.col("EFFECT").str.contains(r'stop')).cast(pl.Int64).alias("STOP"),
        (pl.col("EFFECT").str.contains(r'variant')).cast(pl.Int64).alias("VARIANT"),
        ])
    for c in df.columns:
        if df[c].dtype == pl.Int64:
            df = df.with_columns(
                        df[c].fill_null(0).alias(c)
                    )
            
    mean_columns = ["VAF", "DEPTH", "VAF_*_DEPTH", "MUTATION_LEN"]
    sum_columns = ["ITD", "PTD", "CODON_LOSS", "CODON_GAIN", "CODON_SYNONYMOUS", "STOP", "VARIANT"]
    
    df = df.drop(["CHR", "REF", "ALT", "GENE", "START", "END", "EFFECT", "PROTEIN_CHANGE"])
    
    df = df.group_by("ID").agg([
            *[pl.col(col).mean().alias(col) for col in mean_columns],
            *[pl.col(col).sum().alias(col) for col in sum_columns]
        ])

    return df

def group_molecular_clinical(df_mol, df_clin):
    df = df_clin.join(df_mol, on="ID", how="full")
    for c in df.columns:
        if c == "ID":
            pass
        if df[c].dtype == pl.Int64:
            df = df.with_columns(
                        df[c].fill_null(0).alias(c)
                    )
        if df[c].dtype == pl.Float64:
            df = df.with_columns(
                        df[c].fill_null(df[c].median()).alias(c)
                    )
    return df

SAVE = True

if SAVE:
    Modified_Trainclinical = clinical_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Trainclinical.csv")
    Modified_Trainmolecular = molecular_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Trainmolecular.csv")
    Modified_Ytrain = target_data_engineering("Desktop/QubeChallenge/RawData/TrainDataset/Ytrain.csv")

    Modified_Testclinical = clinical_data_engineering("Desktop/QubeChallenge/RawData/TestDataset/Testclinical.csv")
    Modified_Testmolecular = molecular_data_engineering("Desktop/QubeChallenge/RawData/TestDataset/Testmolecular.csv", traindata=False)

    Traindataset = group_molecular_clinical(Modified_Trainmolecular, Modified_Trainclinical)
    Testdataset = group_molecular_clinical(Modified_Testmolecular, Modified_Testclinical)

    combined = Traindataset.with_columns([Modified_Ytrain["OS_STATUS"], Modified_Ytrain["OS_YEARS"]])
    filtered = combined.filter(~pl.col("OS_YEARS").is_null())

    Traindataset = filtered.select([col for col in filtered.columns if col not in ["ID", "OS_YEARS", "OS_STATUS"]])
    Modified_Ytrain = filtered.select([col for col in filtered.columns if col in ["OS_YEARS", "OS_STATUS"]])

    Traindataset.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
    Modified_Ytrain.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

    Testdataset.write_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testdataset.csv")