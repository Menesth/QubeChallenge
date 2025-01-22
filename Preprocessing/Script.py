import polars as pl

def preprocessing(clinicalpath, molecularpath):
    clinicaldf = pl.read_csv(clinicalpath)
    clinicaldf = clinicaldf.drop(["CENTER"])
    clinicaldf = clinicaldf.with_columns(clinicaldf["MONOCYTES"].cast(pl.Float64).alias("MONOCYTES"))
    moleculardf = pl.read_csv(molecularpath, schema_overrides={"CHR": pl.Utf8})

    # Handling missing values in numerical columns
    for c in clinicaldf.columns:
        if clinicaldf[c].dtype == pl.Float64:
            clinicaldf = clinicaldf.with_columns(clinicaldf[c].fill_nan(clinicaldf[c].median()).alias(c))
            clinicaldf = clinicaldf.with_columns(clinicaldf[c].fill_null(clinicaldf[c].median()).alias(c))
    
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.Float64:
            moleculardf = moleculardf.with_columns(moleculardf[c].fill_nan(moleculardf[c].median()).alias(c))
            moleculardf = moleculardf.with_columns(moleculardf[c].fill_null(moleculardf[c].median()).alias(c))

    # Handling missing values in categorical columns
    for c in clinicaldf.columns:
        if clinicaldf[c].dtype == pl.String:
            clinicaldf = clinicaldf.with_columns(clinicaldf[c].fill_null("unknown").alias(c))
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.String:
            moleculardf = moleculardf.with_columns(moleculardf[c].fill_null("unknown").alias(c))

    # Frequency encoding for high-cardinality categorical columns
    clinicaldf = clinicaldf.with_columns(((pl.len().over("CYTOGENETICS")) / (pl.len())).alias("CYTOGENETICS"))
    for c in ["PROTEIN_CHANGE", "GENE", "ALT", "REF"]:
        moleculardf = moleculardf.with_columns(((pl.len().over(c)) / (pl.len())).alias(c))

    # Frequency encoding for low-cardinality categorical columns
    def one_hot_encoding_(df, column_to_one_hot, possible_strings):
        for s in possible_strings:
            df = df.with_columns((df[column_to_one_hot] == s).cast(pl.Int64).alias(f"{column_to_one_hot}_{s}"))
        df = df.drop(column_to_one_hot)
        return df
    
    moleculardf = one_hot_encoding_(moleculardf, "CHR", ["X"] + [f"{i}" for i in range(1, 23)])
    moleculardf = one_hot_encoding_(moleculardf, "EFFECT", ["ITD", "PTD", "stop_lost", "stop_gained", "inframe_codon_loss", "inframe_codon_gain", "frameshift_variant", "unknown", "non_synonymous_codon"])

    # Groupy "ID" the molecular df
    moleculardf = moleculardf.group_by("ID").agg(
            *[pl.col(c).mean().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Float64],
            *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Int64]
            )

    # Join clinical and molecular df and handle missing values
    outdf = clinicaldf.join(moleculardf, on="ID", how="left")
    for c in outdf.columns:
        if outdf[c].dtype == pl.Float64:
            outdf = outdf.with_columns(outdf[c].fill_nan(outdf[c].median()).alias(c))
            outdf = outdf.with_columns(outdf[c].fill_null(outdf[c].median()).alias(c))
        elif outdf[c].dtype == pl.Int64:
            outdf = outdf.with_columns(outdf[c].fill_nan(0).alias(c))
            outdf = outdf.with_columns(outdf[c].fill_null(0).alias(c))
        else:
            pass

    return outdf