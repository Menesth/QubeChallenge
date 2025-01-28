import polars as pl

def clinicaldf_preprocessing(clinicalpath):
    clinicaldf = pl.read_csv(clinicalpath)
    clinicaldf = clinicaldf.with_columns(clinicaldf["MONOCYTES"].cast(pl.Float64).alias("MONOCYTES"))
    clinicaldf = clinicaldf.drop(["CENTER", "CYTOGENETICS"])

    for c in clinicaldf.columns:
        if clinicaldf[c].dtype==pl.Float64:
            clinicaldf = clinicaldf.with_columns(((clinicaldf[c] - clinicaldf[c].mean()) / clinicaldf[c].std()).alias(c))
            clinicaldf = clinicaldf.with_columns((clinicaldf[c].fill_null(0.0)).alias(c))
    return clinicaldf

def moleculardf_preprocessing(molecularpath):
    moleculardf = pl.read_csv(molecularpath, schema_overrides={"CHR": pl.Utf8})
    moleculardf = moleculardf.drop(["CHR", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT"])
    
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.Float64:
            moleculardf = moleculardf.with_columns(((moleculardf[c] - moleculardf[c].mean()) / moleculardf[c].std()).alias(c))
            moleculardf = moleculardf.with_columns((moleculardf[c].fill_null(0.0)).alias(c))

    moleculardf = moleculardf.group_by("ID").agg(
        *[pl.col(c).mean().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Float64]
        )

    return moleculardf

def get_dataset(clinicalpath, molecularpath):
    clinicaldf = clinicaldf_preprocessing(clinicalpath)
    moleculardf = moleculardf_preprocessing(molecularpath)
    joint_df = clinicaldf.join(moleculardf, on="ID", how="left")

    for c in joint_df.columns:
        if joint_df[c].dtype == pl.Float64:
            joint_df = joint_df.with_columns(((joint_df[c] - joint_df[c].mean()) / joint_df[c].std()).alias(c))
            joint_df = joint_df.with_columns((joint_df[c].fill_null(0.0)).alias(c))
        elif joint_df[c].dtype==pl.Int64:
            joint_df = joint_df.with_columns((joint_df[c].fill_null(0)).alias(c))
        else:
            pass
    return joint_df