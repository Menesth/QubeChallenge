import polars as pl

def one_hot_encoding_(df, column_to_one_hot, possible_strings):
    for s in possible_strings:
        df = df.with_columns((df[column_to_one_hot] == s).cast(pl.Int64).alias(f"{column_to_one_hot}_{s}"))
    df = df.drop(column_to_one_hot)
    return df

def clinicaldf_preprocessing(clinicalpath):
    clinicaldf = pl.read_csv(clinicalpath)
    clinicaldf = clinicaldf.with_columns(clinicaldf["MONOCYTES"].cast(pl.Float64).alias("MONOCYTES"))
    clinicaldf = clinicaldf.drop(["CENTER", "CYTOGENETICS"])
    for c in clinicaldf.columns:
        if clinicaldf[c].dtype==pl.Float64:
            clinicaldf = clinicaldf.with_columns((clinicaldf[c].fill_null(clinicaldf[c].median())).alias(c))
    
    return clinicaldf

def moleculardf_preprocessing(molecularpath):
    moleculardf = pl.read_csv(molecularpath, schema_overrides={"CHR": pl.Utf8})
    moleculardf = moleculardf.with_columns((moleculardf["END"] - moleculardf["START"]).cast(pl.Float64).alias("MUT_LEN"))
    moleculardf = moleculardf.drop(["CHR", "START", "END", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT"])
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.Float64:
            moleculardf = moleculardf.with_columns((moleculardf[c].fill_null(moleculardf[c].median())).alias(c))
    
    moleculardf = moleculardf.group_by("ID").agg(
        *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Float64]
        )

    return moleculardf

def get_dataset(clinicalpath, molecularpath):
    clinicaldf = clinicaldf_preprocessing(clinicalpath)
    moleculardf = moleculardf_preprocessing(molecularpath)
    joint_df = clinicaldf.join(moleculardf, on="ID", how="left")
    for c in joint_df.columns:
        if joint_df[c].dtype == pl.Float64:
            joint_df = joint_df.with_columns((joint_df[c].fill_null(joint_df[c].median())).alias(c))
            joint_df = joint_df.with_columns((joint_df[c].fill_nan(joint_df[c].median())).alias(c))
    return joint_df