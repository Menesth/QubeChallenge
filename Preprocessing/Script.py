import polars as pl

def one_hot_encoding_(df, column_to_one_hot, possible_strings):
    for s in possible_strings:
        df = df.with_columns((df[column_to_one_hot] == s).cast(pl.Int64).alias(f"{column_to_one_hot}_{s}"))
    df = df.drop(column_to_one_hot)
    return df

def clinicaldf_preprocessing(clinicalpath):
    clinicaldf = pl.read_csv(clinicalpath)
    clinicaldf = clinicaldf.drop(["CENTER"])
    clinicaldf = clinicaldf.with_columns(clinicaldf["MONOCYTES"].cast(pl.Float64).alias("MONOCYTES"))

    # Encoding CYTOGENETICS
    clinicaldf = clinicaldf.with_columns((pl.col("CYTOGENETICS").str.contains(r"46,(xy|xx|XX|XY)").cast(pl.Int64)).alias("Normal_count"))
    clinicaldf = clinicaldf.with_columns(
    pl.when(pl.col("CYTOGENETICS").str.contains(r"abnormal|del|iso|inv|add|order|dert|mar|der|complex|dic|inc|idem|t"))
    .then(pl.col("CYTOGENETICS").str.len_chars()).otherwise(0).cast(pl.Int64).alias("Abnormality_Length"))

    clinicaldf = clinicaldf.drop(["CYTOGENETICS"])

    for c in clinicaldf.columns:
        if clinicaldf[c].dtype==pl.Float64:
            clinicaldf = clinicaldf.with_columns((clinicaldf[c].fill_null(clinicaldf[c].median())).alias(c))

    return clinicaldf

def moleculardf_preprocessing(molecularpath):
    moleculardf = pl.read_csv(molecularpath, schema_overrides={"CHR": pl.Utf8})

    # create features
    moleculardf = moleculardf.with_columns((moleculardf["END"] - moleculardf["START"]).cast(pl.Int64).alias("MUT_LEN"))
    #moleculardf = moleculardf.drop(["END", "START"])

    moleculardf = moleculardf.with_columns((moleculardf["DEPTH"] * moleculardf["VAF"]).alias("DEPTH*VAF"))

    # one hot encoding for "EFFECT"
    moleculardf = one_hot_encoding_(moleculardf, "EFFECT", [r"ITD|PTD", r"stop_lost|inframe_codon_loss", r"stop_gained|inframe_codon_gain", "frameshift_variant", "non_synonymous_codon"])

    # frequency encoding for "REF", "ALT", "GENE"
    moleculardf = moleculardf.with_columns((moleculardf["REF"] + "," + moleculardf["ALT"]).cast(pl.String).alias("REF_ALT"))
    moleculardf = moleculardf.with_columns((pl.len().over("REF_ALT") / pl.len()).alias("REF_ALT"))
    moleculardf = moleculardf.with_columns((pl.len().over("GENE") / pl.len()).alias("GENE"))
    moleculardf = moleculardf.drop(["REF", "ALT", "PROTEIN_CHANGE", "CHR"])

    moleculardf = moleculardf.group_by("ID").agg(
        *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Float64],
        *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Int64]
        )

    # Handling missing values
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.Float64:
            moleculardf = moleculardf.with_columns((moleculardf[c].fill_null(moleculardf[c].median())).alias(c))

    return moleculardf

def get_dataset(clinicalpath, molecularpath):
    clinicaldf = clinicaldf_preprocessing(clinicalpath)
    moleculardf = moleculardf_preprocessing(molecularpath)
    joint_df = clinicaldf.join(moleculardf, on="ID", how="left")
    return joint_df