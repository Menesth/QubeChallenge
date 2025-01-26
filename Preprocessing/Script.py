import polars as pl

def one_hot_encoding_(df, column_to_one_hot, possible_strings):
    for s in possible_strings:
        df = df.with_columns((df[column_to_one_hot] == s).cast(pl.Int64).alias(f"{column_to_one_hot}_{s}"))
    df = df.drop(column_to_one_hot)
    return df

def clinicaldf_preprocessing(clinicalpath):
    clinicaldf = pl.read_csv(clinicalpath)
    clinicaldf = clinicaldf.with_columns(clinicaldf["MONOCYTES"].cast(pl.Float64).alias("MONOCYTES"))
    
    clinicaldf = clinicaldf.with_columns([pl.when(pl.col("CYTOGENETICS").str.contains(r'^(?:46,xx(?:\[\d+\])?|46,xy(?:\[\d+\])?)$')).then(pl.lit("Normal")).otherwise(pl.col("CYTOGENETICS")).alias("CYTOGENETICS")])
    clinicaldf = clinicaldf.with_columns([pl.when(pl.col("CYTOGENETICS").str.contains("plex")).then(pl.lit("Complex")).otherwise(pl.col("CYTOGENETICS")).alias("CYTOGENETICS")])

    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains("Complex").cast(pl.Int64)).alias("is_complex"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains("Normal").cast(pl.Int64)).alias("is_normal"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains("der").cast(pl.Int64)).alias("der"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains("t").cast(pl.Int64)).alias("t"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains(r"-2|\+2").cast(pl.Int64)).alias("2"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains(r"-3|\+3").cast(pl.Int64)).alias("3"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains(r"-6|\+6").cast(pl.Int64)).alias("6"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains(r"-7|\+7").cast(pl.Int64)).alias("7"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.contains(r"-22|\+22").cast(pl.Int64)).alias("22"))

    clinicaldf = clinicaldf.with_columns((clinicaldf["CYTOGENETICS"].str.count_matches(",").cast(pl.Int64)).alias("num_abnormalities"))

    clinicaldf = clinicaldf.drop(["CENTER", "CYTOGENETICS"])

    clinicaldf = clinicaldf.with_columns((clinicaldf["ANC"] / (clinicaldf["ANC"] + clinicaldf["HB"])).alias("ANC/HB"))
    clinicaldf = clinicaldf.with_columns((clinicaldf["PLT"] / (clinicaldf["PLT"] + clinicaldf["HB"])).alias("PLT/HB"))

    for c in clinicaldf.columns:
        if clinicaldf[c].dtype==pl.Float64:
            clinicaldf = clinicaldf.with_columns((clinicaldf[c].fill_null(clinicaldf[c].median())).alias(c))
        elif clinicaldf[c].dtype==pl.Int64:
            clinicaldf = clinicaldf.with_columns((clinicaldf[c].fill_null(0)).alias(c))
        else:
            pass
    return clinicaldf

def moleculardf_preprocessing(molecularpath):
    moleculardf = pl.read_csv(molecularpath, schema_overrides={"CHR": pl.Utf8})
    
    moleculardf = moleculardf.with_columns((moleculardf["END"] - moleculardf["START"]).cast(pl.Float64).alias("MUT_LEN"))
    
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("X").cast(pl.Int64)).alias("CHRX"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("1").cast(pl.Int64)).alias("CHR1"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("2").cast(pl.Int64)).alias("CHR2"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("7").cast(pl.Int64)).alias("CHR7"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("9").cast(pl.Int64)).alias("CHR9"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("20").cast(pl.Int64)).alias("CHR20"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("21").cast(pl.Int64)).alias("CHR21"))
    moleculardf = moleculardf.with_columns((moleculardf["CHR"].str.contains("22").cast(pl.Int64)).alias("CHR22"))
    
    moleculardf = moleculardf.with_columns((moleculardf["ALT"].str.contains("A").cast(pl.Int64)).alias("ALT_A"))
    
    moleculardf = moleculardf.with_columns((moleculardf["EFFECT"].str.contains("stop").cast(pl.Int64)).alias("S"))
    moleculardf = moleculardf.with_columns((moleculardf["EFFECT"].str.contains("frameshift_variant").cast(pl.Int64)).alias("FV"))
    moleculardf = moleculardf.with_columns((moleculardf["EFFECT"].str.contains("non_synonymous_codon").cast(pl.Int64)).alias("NSC"))
    moleculardf = moleculardf.with_columns((moleculardf["EFFECT"].str.contains("inframe_codon_gain|inframe_codon_loss").cast(pl.Int64)).alias("IC"))
    moleculardf = moleculardf.with_columns((moleculardf["EFFECT"].str.contains("ITD|PTD").cast(pl.Int64)).alias("TD"))

    moleculardf = moleculardf.with_columns((moleculardf["GENE"].str.contains("FLT").cast(pl.Int64)).alias("FLT"))
    moleculardf = moleculardf.with_columns((moleculardf["GENE"].str.contains("TET").cast(pl.Int64)).alias("TET"))
    moleculardf = moleculardf.with_columns((moleculardf["GENE"].str.contains("JAK").cast(pl.Int64)).alias("JAK"))
    moleculardf = moleculardf.with_columns((moleculardf["GENE"].str.contains("ZRSR").cast(pl.Int64)).alias("ZRSR"))
    moleculardf = moleculardf.with_columns((moleculardf["GENE"].str.contains("TET").cast(pl.Int64)).alias("TET"))

    moleculardf = moleculardf.with_columns((moleculardf["PROTEIN_CHANGE"].str.contains("Q").cast(pl.Int64)).alias("Q"))
    moleculardf = moleculardf.with_columns((moleculardf["PROTEIN_CHANGE"].str.contains("H").cast(pl.Int64)).alias("H"))
    moleculardf = moleculardf.with_columns((moleculardf["PROTEIN_CHANGE"].str.contains("L").cast(pl.Int64)).alias("L"))

    moleculardf = moleculardf.drop(["CHR", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT"])
    
    for c in moleculardf.columns:
        if moleculardf[c].dtype == pl.Float64:
            moleculardf = moleculardf.with_columns((moleculardf[c].fill_null(moleculardf[c].median())).alias(c))
        elif moleculardf[c].dtype==pl.Int64:
            moleculardf = moleculardf.with_columns((moleculardf[c].fill_null(0)).alias(c))
        else:
            pass

    moleculardf = moleculardf.group_by("ID").agg(
        *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Float64],
        *[pl.col(c).max().alias(c) for c in moleculardf.columns if moleculardf[c].dtype == pl.Int64]
        )

    return moleculardf

def get_dataset(clinicalpath, molecularpath):
    clinicaldf = clinicaldf_preprocessing(clinicalpath)
    moleculardf = moleculardf_preprocessing(molecularpath)
    joint_df = clinicaldf.join(moleculardf, on="ID", how="left")

    joint_df = joint_df.with_columns((joint_df["VAF"] * joint_df["PLT"]).alias("VAF*PLT"))
    joint_df = joint_df.with_columns((joint_df["VAF"] * joint_df["WBC"]).alias("VAF*WBC"))
    joint_df = joint_df.with_columns((joint_df["VAF"] * joint_df["HB"]).alias("VAF*HB"))
    joint_df = joint_df.with_columns((joint_df["DEPTH"] * joint_df["BM_BLAST"]).alias("DEPTH*BM_BLAST"))

    joint_df = joint_df.with_columns((joint_df["ANC"] / (joint_df["ANC"] + joint_df["DEPTH"])).alias("ANC/DEPTH"))
    joint_df = joint_df.with_columns((joint_df["PLT"] / (joint_df["PLT"] + joint_df["VAF"])).alias("PLT/VAF"))
    joint_df = joint_df.with_columns((joint_df["PLT"] / (joint_df["PLT"] + joint_df["DEPTH"])).alias("PLT/DEPTH"))

    for c in joint_df.columns:
        if joint_df[c].dtype == pl.Float64:
            joint_df = joint_df.with_columns((joint_df[c].fill_null(joint_df[c].median())).alias(c))
        elif joint_df[c].dtype==pl.Int64:
            joint_df = joint_df.with_columns((joint_df[c].fill_null(0)).alias(c))
        else:
            pass
    return joint_df