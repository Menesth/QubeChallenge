import polars as pl

def target_data_engineering(path):

    df = pl.read_csv(path)
    df = df.with_columns(df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID"))
    df = df.sort("ID")
    return df

def clinical_data_engineering(path):

    df = pl.read_csv(path)
    df = df.with_columns(df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID"))
    df = df.sort("ID")
    df = df.drop(["CENTER"])
    df = df.with_columns(df["MONOCYTES"].cast(pl.Float64))

    df = df.with_columns([
    (pl.col("CYTOGENETICS") == "Normal").alias("Normal").cast(pl.Int64),
    (pl.col("CYTOGENETICS").str.contains("Abnormal|Not evaluated")).alias("Abnormal").cast(pl.Int64),
    (pl.col("CYTOGENETICS").str.contains("plex")).alias("Complex").cast(pl.Int64),
    pl.when(
        (pl.col("CYTOGENETICS") != "Normal") &
        ~pl.col("CYTOGENETICS").str.contains("Abnormal|Not evaluated|plex")
    ).then(pl.col("CYTOGENETICS").str.len_chars())
    .otherwise(0)
    .cast(pl.Int64)
    .alias("CYTOGENETICS_LEN")
    ])
    df = df.drop(["CYTOGENETICS"])
    #df = df.with_columns([
    #    (pl.col("CYTOGENETICS").str.contains('Abnormal')).cast(pl.Int64).alias("Abnormal"),
    #    (pl.col("CYTOGENETICS").str.contains('Normal')).cast(pl.Int64).alias("Normal"),
    #    (pl.col("CYTOGENETICS").str.contains('del')).cast(pl.Int64).alias("del"),
    #    (pl.col("CYTOGENETICS").str.contains('dic')).cast(pl.Int64).alias("dic"),
    #    (pl.col("CYTOGENETICS").str.contains('der')).cast(pl.Int64).alias("der"),
    #    (pl.col("CYTOGENETICS").str.contains('add')).cast(pl.Int64).alias("add"),
    #    (pl.col("CYTOGENETICS").str.contains('dup')).cast(pl.Int64).alias("dup"),
    #    (pl.col("CYTOGENETICS").str.contains('t')).cast(pl.Int64).alias("tr"),
    #    (pl.col("CYTOGENETICS").str.contains('inv')).cast(pl.Int64).alias("inv"),
    #    (pl.col("CYTOGENETICS").str.contains('ins')).cast(pl.Int64).alias("ins"),
    #    (pl.col("CYTOGENETICS").str.contains(r'i')).cast(pl.Int64).alias("iso"),
    #    (pl.col("CYTOGENETICS").str.contains('mar')).cast(pl.Int64).alias("mar"),
    #    (pl.col("CYTOGENETICS").str.contains('r')).cast(pl.Int64).alias("ring"),
    #    (pl.col("CYTOGENETICS").str.contains(r'\-')).cast(pl.Int64).alias("subs_"),
    #    (pl.col("CYTOGENETICS").str.contains(r'\+')).cast(pl.Int64).alias("add_"),
    #    (pl.col("CYTOGENETICS").str.contains(r'\*')).cast(pl.Int64).alias("mult_"),
    #    (pl.col("CYTOGENETICS").str.contains(r'\/')).cast(pl.Int64).alias("div_"),
    #    (pl.col("CYTOGENETICS").str.contains('inc')).cast(pl.Int64).alias("inc")
    #    ])
    #df = df.with_columns((pl.col("add") + pl.col("dic") + pl.col("div_") + pl.col("mult_") + pl.col("inc") + pl.col("subs_") + pl.col("ring") + pl.col("der") + pl.col("add_") + pl.col("mar") + pl.col("iso") + pl.col("ins") + pl.col("inv") + pl.col("tr") + pl.col("dup") + pl.col("del")).alias("CYTOGEN_COMPLEXITY"))
    #df = df.drop(["CYTOGENETICS", "dic", "add", "div_", "inc", "mult_", "add_", "subs_", "ring", "mar", "iso", "ins", "inv", "tr", "dup", "del", "der"])

    columns_to_mult = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    #for c1 in columns_to_mult:
    #    df = df.with_columns(((1 + df[c1]).log()).alias(f"log({c1})"))
    #    for c2 in columns_to_mult:
    #        if c1 != c2:
    #            df = df.with_columns((df[c1] * df[c2]).alias(f"{c1}*{c2}"))

    cont_columns = [c for c in df.columns if df[c].dtype == pl.Float64 or c == "CYTOGEN_COMPLEXITY"]
    disc_columns = [c for c in df.columns if df[c].dtype == pl.Int64 and c != "ID" and c != "CYTOGEN_COMPLEXITY"]

    for c in df.columns:
        if c in cont_columns:
            #df = df.with_columns(((df[c] - df[c].min()) / (df[c].max() - df[c].min() )).alias(c))
            df = df.with_columns(df[c].fill_null(df[c].median()).alias(c))
            df = df.with_columns(df[c].fill_nan(df[c].median()).alias(c))
        elif c in disc_columns:
            df = df.with_columns(df[c].fill_null(0).alias(c))
            df = df.with_columns(df[c].fill_nan(0).alias(c))
        else:
            pass
    return df

def molecular_data_engineering(path, traindata=True):

    if traindata:
        df = pl.read_csv(path)
        RANGE = 132729 + 1
    else:
        df = pl.read_csv(path, schema_overrides={"CHR": pl.Utf8})
        RANGE = 1193 + 1
    df = df.with_columns(df["ID"].str.extract(r'\d+', 0).cast(pl.Int64).alias("ID"))
    df = df.sort("ID")
    df = df.with_columns((df['END'] - df["START"] + 1).alias("MUTATION_LEN"))
    df = df.drop(["START", "END"])
    df = df.with_columns([
        #(pl.col("CHR").str.contains(r'X')).cast(pl.Int64).alias("CHRX"),
        (pl.col("CHR").str.contains(r'X|1|2|4|6|9|10|12|13|14|15|16|18|19|20|22')).cast(pl.Int64).alias("CHRother"),
        (pl.col("CHR").str.contains(r'3')).cast(pl.Int64).alias("CHR3"),
        (pl.col("CHR").str.contains(r'5')).cast(pl.Int64).alias("CHR5"),
        (pl.col("CHR").str.contains(r'7')).cast(pl.Int64).alias("CHR7"),
        (pl.col("CHR").str.contains(r'8')).cast(pl.Int64).alias("CHR8"),
        (pl.col("CHR").str.contains(r'11')).cast(pl.Int64).alias("CHR11"),
        (pl.col("CHR").str.contains(r'17')).cast(pl.Int64).alias("CHR17"),
        (pl.col("CHR").str.contains(r'21')).cast(pl.Int64).alias("CHR21"),
        ])
    df = df.drop("CHR")
    df = df.with_columns([
        (pl.col("EFFECT").str.contains(r'non_synonymous_codon')).cast(pl.Int64).alias("non_codon"),
        (pl.col("EFFECT").str.contains(r'ITD|PTD')).cast(pl.Int64).alias("I_or_D_TP"),
        (pl.col("EFFECT").str.contains(r'inframe_codon_gain|inframe_codon_loss')).cast(pl.Int64).alias("inframe_codon"),
        (pl.col("EFFECT").str.contains(r'stop_lost|stop_gained')).cast(pl.Int64).alias("stop"),
        (pl.col("EFFECT").str.contains(r'frameshift_variant')).cast(pl.Int64).alias("shift"),
        ])
    df = df.drop("EFFECT")
    df = df.with_columns([
        (pl.col("GENE").str.contains(r'TP53|RB1|CDKN2A')).cast(pl.Int64).alias("TSG"),
        (pl.col("GENE").str.contains(r'FLT3|RAS|KRAS|NRAS|KIT|MYC')).cast(pl.Int64).alias("ONCO"),
        (pl.col("GENE").str.contains(r'DNMT3A|TET2|IDH1|IDH2|ASXL1')).cast(pl.Int64).alias("EPIG"),
        (pl.col("GENE").str.contains(r'BRCA1|BRCA2|ATM')).cast(pl.Int64).alias("DNADRG"),
        (pl.col("GENE").str.contains(r'SF3B1|SRSF2|U2AF1')).cast(pl.Int64).alias("SF"),
        (pl.col("GENE").str.contains(r'RUNX1|CEBPA|GATA2')).cast(pl.Int64).alias("TF"),
        (pl.col("GENE").str.contains(r'JAK2|CSF3R')).cast(pl.Int64).alias("STG"),
        (pl.col("GENE").str.contains(r'STAG2|RAD21|SMC3|SMC1A')).cast(pl.Int64).alias("CCG"),
        ])
    df = df.drop("GENE")

    cont_columns = ["VAF", "DEPTH", "MUTATION_LEN"]
    disc_columns = [c for c in df.columns if df[c].dtype == pl.Int64 and c != "ID"]

    df = df.with_columns((df["VAF"] * df["DEPTH"]).alias("VAF*DEPTH"))
    df = df.with_columns((df["VAF"] * df["MUTATION_LEN"]).alias("VAF*MUTATION_LEN"))
    df = df.with_columns((df["DEPTH"] * df["MUTATION_LEN"]).alias("DEPTH*MUTATION_LEN"))

    cont_columns = ["VAF", "DEPTH", "MUTATION_LEN", "VAF*DEPTH", "VAF*MUTATION_LEN", "DEPTH*MUTATION_LEN"]

    df = df.group_by("ID").agg([
            *[pl.col(col).mean().alias(col) for col in cont_columns],
            *[pl.col(col).sum().alias(col) for col in disc_columns]
            ])
    
    for c in df.columns:
        if c in disc_columns:
            df = df.with_columns(df[c].fill_null(0).alias(c))
            df = df.with_columns(df[c].fill_nan(0).alias(c))
        elif c in cont_columns:
            #df = df.with_columns(((df[c] - df[c].min()) / (df[c].max() - df[c].min() )).alias(c))
            df = df.with_columns(df[c].fill_null(df[c].median()).alias(c))
            df = df.with_columns(df[c].fill_nan(df[c].median()).alias(c))
        else:
            pass
    return df

def group_molecular_clinical(df_mol, df_clin):
    df = df_clin.join(df_mol, on="ID", how="left")
    for c in df.columns:
        if c == "ID":
            pass
        if df[c].dtype == pl.Int64:
            df = df.with_columns(df[c].fill_null(0).alias(c))
            df = df.with_columns(df[c].fill_nan(0).alias(c))
        if df[c].dtype == pl.Float64:
            df = df.with_columns(df[c].fill_null(df[c].median()).alias(c))
            df = df.with_columns(df[c].fill_nan(df[c].median()).alias(c))
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

    Traindataset = filtered.select([col for col in filtered.columns if col not in ["OS_YEARS", "OS_STATUS"]])
    Modified_Ytrain = filtered.select([col for col in filtered.columns if col in ["ID", "OS_YEARS", "OS_STATUS"]])

    Traindataset.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Traindataset.csv")
    Modified_Ytrain.write_csv("Desktop/QubeChallenge/ModifiedData/TrainDataset/Ytrain.csv")

    Testdataset.write_csv("Desktop/QubeChallenge/ModifiedData/TestDataset/Testdataset.csv")