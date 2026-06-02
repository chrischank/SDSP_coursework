import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore and Predict the Spatio-Temporal relationship of Scotland based on SIMD and Housing prices
    """)
    return


@app.cell
def _():
    ###############################
    # Notebook for SPDS Assessment#
    # Maintainer: Christopher Chan#
    # Version: 0.0.1              #
    # Date: 2026-06-02            #
    ###############################

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from typing import Literal

    DATA_RAW = Path("../data/01_raw")
    Path("../data/02_intermediate").mkdir(parents=True, exist_ok=True)
    Path("../data/03_feature").mkdir(parents=True, exist_ok=True)
    DATA_INTERMEDIATE = Path("../data/02_intermediate")
    DATA_FEATURE = Path("../data/02_feature")
    return DATA_RAW, Literal, Path, gpd, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ### Data Preprocesseing
    #### SIMD
    1. Roll up SIMD to council area
    2. Calculate rank -> Decile
    3. Calculate Diffs

    #### ROS
    1. Adjust median house prices for mortgage sheet to 2012 base price (inflation adjustment)
    2. Compute volume weighted median for corresponding temporal alignment with SIMD
    3. Calculate Diffs
    """)
    return


@app.cell
def _(DATA_RAW, gpd, np, pd):
    # SIMD Preprocessing

    simd_geom= gpd.read_file(f"{DATA_RAW}/simd2012_withgeog/DZ_2011_EoR_Scotland.shp")

    simd2012_cols = ["Data Zone", "Local Authority Name", "Intermediate Geography name",
                   "Health domain 2012 rank", "Income domain 2012 rank", "Employment domain 2012 rank",
                   "Education, Skills and Training domain 2012 rank", "Housing domain rank 2004, 2006, 2009 & 2012",
                   "Geographic Access domain 2012 rank", "SIMD Crime 2012 rank"]
    simd2016_cols = ["Data_Zone", "Intermediate_Zone", "Council_area",
                                "Income_Domain_2016_Rank", "Employment_Domain_2016_Rank", 
                                "Health_Domain_2016_Rank", "Education_Domain_2016_Rank", 
                                "Geographic_Access_Domain_2016_Rank", "Crime_Domain_2016_Rank", 
                                "Housing_Domain_2016_Rank"]
    simd2020_cols = ["Data_Zone", "Intermediate_Zone", "Council_area",
                                "SIMD2020v2_Income_Domain_Rank", "SIMD2020_Employment_Domain_Rank", 
                                "SIMD2020_Health_Domain_Rank", "SIMD2020_Education_Domain_Rank", 
                                "SIMD2020_Access_Domain_Rank", "SIMD2020_Crime_Domain_Rank",
                                "SIMD2020_Housing_Domain_Rank"]

    assert len(simd2012_cols) == len(simd2016_cols) == len(simd2020_cols), "col counts does not match"


    simd_2012 = pd.read_csv(f"{DATA_RAW}/simd2012_withgeog/simd2012_data_00410767_plusintervals.csv",
        usecols = [*simd2012_cols],
        thousands=",",
        dtype = {d: np.int32 for d in simd2012_cols if d not in ["Data Zone", "Local Authority Name", "Intermediate Geography name"]}
    )

    simd_2012 = simd_geom.merge(simd_2012, left_on="DZ_CODE", right_on="Data Zone", how="left")

    simd_2012.plot("Income domain 2012 rank", legend=True)
    return simd2012_cols, simd2016_cols, simd2020_cols


@app.cell
def _(Literal, Path, gpd, np, pd, simd2012_cols, simd2016_cols, simd2020_cols):
    DIFF_PERIOD = Literal["12-16", "16-20"]

    def simd_preprocessing(
        simd_geom_past: Path, simd_data_past: Path, 
        simd_geom_future: Path, simd_data_future: Path, 
        diff: DIFF_PERIOD
        ) -> gpd.GeoDataFrame:

        def _agg_dict(columns:list) -> dict:
            agg_dict = {
                f"{col}": "sum" for col in columns
            }

            return agg_dict

        if diff == "12-16":
            simd_geom_pastDF = gpd.read_file(simd_geom_past)
            simd_data_pastDF = pd.read_csv(simd_data_past,
                usecols = [*simd2012_cols],
                thousands = ",",
                dtype = {d: np.int32 for d in simd2012_cols if d not in ["Data Zone", "Local Authority Name", "Intermediate Geography name"]}
            )

            simd_geom_futureDF = gpd.read_file(simd_geom_future)
            simd_data_futureDF = pd.read_csv(simd_data_future,
                usecols = [*simd2016_cols],
            )

            simd_pastDF =  simd_geom_pastDF.merge(simd_data_pastDF, left_on="DZ_CODE", right_on="Data Zone", how="left")
            simd_futureDF = simd_geom_futureDF.merge(simd_data_futureDF, left_on="DataZone", right_on="Data_Zone", how="left")

            domain_cols2012 = [col for col in simd2012_cols if col not in ["Data Zone", "Local Authority Name", "Intermediate Geography name"]]
            domain_cols2016 = [col for col in simd2016_cols if col not in ["Data_Zone", "Intermediate_Zone", "Council_area"]]


            # Decile count weighted geometric dissolve to council area
            # I need to do this because some council areas are larger
            # Therefore pure aggregation will result in distorted stats

            # Use a dummy count of 1 for dissolve so I get the count.
            simd_pastDF["polygon_count"] = 1
            DissolveSIMD_pastDF = simd_pastDF.dissolve(
                by="Local Authority Name",
                aggfunc=_agg_dict([*domain_cols2012, "polygon_count"])
                )

            for col in domain_cols2012:
                DissolveSIMD_pastDF[f"weighted_{col}"] = DissolveSIMD_pastDF[f"{col}"] / DissolveSIMD_pastDF["polygon_count"]

        elif diff == "16-20":
            simd_cols_past, simd_cols_future = simd2016_cols, simd2020_cols


        return simd_pastDF, simd_futureDF, DissolveSIMD_pastDF


    return (simd_preprocessing,)


@app.cell
def _(DATA_RAW, Path, simd_preprocessing):
    simd_2012DF, simd_2016DF, DissolveSIMD_2012DF= simd_preprocessing(
        simd_geom_past = Path(f"{DATA_RAW}/simd2012_withgeog/DZ_2011_EoR_Scotland.shp"),
        simd_data_past = Path(f"{DATA_RAW}/simd2012_withgeog/simd2012_data_00410767_plusintervals.csv"),
        simd_geom_future = Path(f"{DATA_RAW}/simd2016_withgeog/sc_dz_11.shp"),
        simd_data_future = Path(f"{DATA_RAW}/simd2016_withgeog/simd2016_withinds.csv"),
        diff = "12-16"
        )
    return DissolveSIMD_2012DF, simd_2012DF, simd_2016DF


@app.cell
def _(DissolveSIMD_2012DF):
    DissolveSIMD_2012DF
    return


@app.cell
def _(DissolveSIMD_2012DF, plt, simd_2012DF, simd_2016DF):
    fig, axes = plt.subplots(1, 3, figsize = (30, 10))
    simd_2012DF.plot(
        "Income domain 2012 rank",
        ax=axes[0],
        legend=True
    )
    simd_2016DF.plot(
        "Income_Domain_2016_Rank",
        ax=axes[1],
        legend=True
    )
    DissolveSIMD_2012DF.plot(
        "weighted_Income domain 2012 rank",
        ax=axes[2],
        legend=True
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
