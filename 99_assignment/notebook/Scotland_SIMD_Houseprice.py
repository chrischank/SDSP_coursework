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
    # Version: 0.0.2              #
    # Date: 2026-06-03            #
    ###############################

    import sys
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as cx

    from pathlib import Path
    from typing import Literal

    sys.path.append("..")
    from src.dictionary import SIMD_DOMAIN_2012, SIMD_DOMAIN_2016, SIMD_DOMAIN_2020

    DATA_RAW = Path("../data/01_raw")
    Path("../data/02_intermediate").mkdir(parents=True, exist_ok=True)
    Path("../data/03_feature").mkdir(parents=True, exist_ok=True)
    DATA_INTERMEDIATE = Path("../data/02_intermediate")
    DATA_FEATURE = Path("../data/03_feature")
    return (
        DATA_INTERMEDIATE,
        DATA_RAW,
        Literal,
        Path,
        SIMD_DOMAIN_2012,
        SIMD_DOMAIN_2016,
        SIMD_DOMAIN_2020,
        cx,
        gpd,
        np,
        pd,
        plt,
    )


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
def _(
    Literal,
    Path,
    SIMD_DOMAIN_2012,
    SIMD_DOMAIN_2016,
    SIMD_DOMAIN_2020,
    gpd,
    np,
    pd,
):
    DIFF_PERIOD = Literal["12-16", "16-20"]

    def simd_preprocessing(
        simd_geom_past: Path,
        simd_data_past: Path,
        simd_geom_future: Path,
        simd_data_future: Path,
        diff: DIFF_PERIOD,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

        def _agg_dict(columns: list) -> dict:
            agg_dict = {f"{col}": "sum" for col in columns}

            return agg_dict

        def _weighted_domain_dissolve(
            gdf: gpd.GeoDataFrame, dissolveby: str, domain_cols: list
        ) -> gpd.GeoDataFrame:
            gdf["polygon_count"] = 1
            dissolve_df = gdf.dissolve(
                by=dissolveby, aggfunc=_agg_dict([*domain_cols, "polygon_count"])
            )

            for col in domain_cols:
                dissolve_df[f"weighted_{col}"] = (
                    dissolve_df[f"{col}"] / dissolve_df["polygon_count"]
                )

            return dissolve_df

        if diff == "12-16":
            simd_geom_pastDF = gpd.read_file(simd_geom_past)
            simd_data_pastDF = pd.read_csv(
                simd_data_past,
                usecols=[*SIMD_DOMAIN_2012.keys()],
                thousands=",",
                dtype={
                    d: np.int32
                    for d in [*SIMD_DOMAIN_2012.keys()]
                    if d
                    not in [
                        "Data Zone",
                        "Local Authority Name",
                        "Intermediate Geography name",
                    ]
                },
            )

            simd_geom_futureDF = gpd.read_file(simd_geom_future)

            # Check projection
            if simd_geom_pastDF.crs != 27700:
                simd_geom_pastDF = simd_geom_pastDF.to_crs(27700)
            if simd_geom_futureDF.crs != 27700:
                simd_geom_futureDF = simd_geom_futureDF.to_crs(27700)

            simd_data_futureDF = pd.read_csv(
                simd_data_future,
                usecols=[*SIMD_DOMAIN_2016.keys()],
            )

            # Rename columns with dictionary
            simd_data_pastDF.rename(columns=SIMD_DOMAIN_2012, inplace=True)
            simd_data_futureDF.rename(columns=SIMD_DOMAIN_2016, inplace=True)

            simd_pastDF = simd_geom_pastDF.merge(
                simd_data_pastDF, left_on="DZ_CODE", right_on="Data_Zone", how="left"
            )
            simd_futureDF = simd_geom_futureDF.merge(
                simd_data_futureDF, left_on="DataZone", right_on="Data_Zone", how="left"
            )

            domain_cols2012 = [
                col
                for col in [*SIMD_DOMAIN_2012.values()]
                if col not in ["Data_Zone", "Council_Area", "Intermediate_Zone"]
            ]
            domain_cols2016 = [
                col
                for col in [*SIMD_DOMAIN_2016.values()]
                if col not in ["Data_Zone", "Council_Area", "Intermediate_Zone"]
            ]

            # Decile count weighted geometric dissolve to council area
            # I need to do this because some council areas are larger
            # Therefore pure aggregation will result in distorted stats

            # Use a dummy count of 1 for dissolve so I get the count.
            DissolveSIMD_pastDF = _weighted_domain_dissolve(
                simd_pastDF, "Council_Area", domain_cols2012
            )
            DissolveSIMD_futureDF = _weighted_domain_dissolve(
                simd_futureDF, "Council_Area", domain_cols2016
            )

        elif diff == "16-20":
            simd_geom_pastDF = gpd.read_file(simd_geom_past)
            simd_data_pastDF = pd.read_csv(
                simd_data_past, usecols=[*SIMD_DOMAIN_2016.keys()]
            )

            simd_geom_futureDF = gpd.read_file(simd_geom_future)
            # Check projection
            if simd_geom_pastDF.crs != 27700:
                simd_geom_pastDF = simd_geom_pastDF.to_crs(27700)
            if simd_geom_futureDF.crs != 27700:
                simd_geom_futureDF = simd_geom_futureDF.to_crs(27700)

            simd_data_futureDF = pd.read_csv(
                simd_data_future, usecols=[*SIMD_DOMAIN_2020.keys()]
            )

            # Rename columns with dictionary
            simd_data_pastDF.rename(columns=SIMD_DOMAIN_2016, inplace=True)
            simd_data_futureDF.rename(columns=SIMD_DOMAIN_2020, inplace=True)

            simd_pastDF = simd_geom_pastDF.merge(
                simd_data_pastDF, left_on="DataZone", right_on="Data_Zone", how="left"
            )
            simd_futureDF = simd_geom_futureDF.merge(
                simd_data_futureDF, left_on="DataZone", right_on="Data_Zone", how="left"
            )

            domain_cols_past = [
                col
                for col in [*SIMD_DOMAIN_2016.values()]
                if col not in ["Data_Zone", "Council_Area", "Intermediate_Zone"]
            ]
            domain_cols_future = [
                col
                for col in [*SIMD_DOMAIN_2020.values()]
                if col not in ["Data_Zone", "Council_Area", "Intermediate_Zone"]
            ]

            # Decile count weighted geometric dissolve
            DissolveSIMD_pastDF = _weighted_domain_dissolve(
                simd_pastDF, "Council_Area", domain_cols_past
            )
            DissolveSIMD_futureDF = _weighted_domain_dissolve(
                simd_futureDF, "Council_Area", domain_cols_future
            )

        weighted_pastcols = [
            col for col in DissolveSIMD_pastDF.columns if col.startswith("weighted_")
        ]
        weighted_futurecols = [
            col for col in DissolveSIMD_futureDF.columns if col.startswith("weighted_")
        ]

        dissolve_pastcols = ["geometry"] + weighted_pastcols
        dissolve_futurecols = ["geometry"] + weighted_futurecols
        DissolveSIMD_pastDF = DissolveSIMD_pastDF[dissolve_pastcols].reset_index()
        DissolveSIMD_futureDF = DissolveSIMD_futureDF[dissolve_futurecols].reset_index()

        for x, y in zip(weighted_pastcols, weighted_futurecols):
            DissolveSIMD_pastDF[f"{x}_decile"] = pd.qcut(
                DissolveSIMD_pastDF[x], 10, labels=False
            )
            DissolveSIMD_futureDF[f"{y}_decile"] = pd.qcut(
                DissolveSIMD_futureDF[y], 10, labels=False
            )

        return simd_pastDF, simd_futureDF, DissolveSIMD_pastDF, DissolveSIMD_futureDF

    return (simd_preprocessing,)


@app.cell
def _(DATA_INTERMEDIATE, DATA_RAW, Path, simd_preprocessing):
    simd_2012DF, simd_2016DF, DissolveSIMD_2012DF, DissolveSIMD_2016DF = (
        simd_preprocessing(
            simd_geom_past=Path(
                f"{DATA_RAW}/simd2012_withgeog/DZ_2011_EoR_Scotland.shp"
            ),
            simd_data_past=Path(
                f"{DATA_RAW}/simd2012_withgeog/simd2012_data_00410767_plusintervals.csv"
            ),
            simd_geom_future=Path(f"{DATA_RAW}/simd2016_withgeog/sc_dz_11.shp"),
            simd_data_future=Path(
                f"{DATA_RAW}/simd2016_withgeog/simd2016_withinds.csv"
            ),
            diff="12-16",
        )
    )

    simd_2016DF, simd_2020DF, DissolveSIMD_2016DF, DissolveSIMD_2020DF = (
        simd_preprocessing(
            simd_geom_past=Path(f"{DATA_RAW}/simd2016_withgeog/sc_dz_11.shp"),
            simd_data_past=Path(f"{DATA_RAW}/simd2016_withgeog/simd2016_withinds.csv"),
            simd_geom_future=Path(f"{DATA_RAW}/simd2020_withgeog/sc_dz_11.shp"),
            simd_data_future=Path(
                f"{DATA_RAW}/simd2020_withgeog/simd2020_withinds.csv"
            ),
            diff="16-20",
        )
    )

    simd_2012DF.to_file(f"{DATA_INTERMEDIATE}/simd_2012.geojson", driver="GeoJSON")
    simd_2016DF.to_file(f"{DATA_INTERMEDIATE}/simd_2016.geojson", driver="GeoJSON")

    DissolveSIMD_2012DF.to_file(
        f"{DATA_INTERMEDIATE}/DissolveSIMD_2012.geojson", driver="GeoJSON"
    )
    DissolveSIMD_2016DF.to_file(
        f"{DATA_INTERMEDIATE}/DissolveSIMD_2016.geojson", driver="GeoJSON"
    )
    DissolveSIMD_2020DF.to_file(
        f"{DATA_INTERMEDIATE}/DissolveSIMD_2020.geojson", driver="GeoJSON"
    )
    return (
        DissolveSIMD_2012DF,
        DissolveSIMD_2016DF,
        DissolveSIMD_2020DF,
        simd_2012DF,
        simd_2016DF,
        simd_2020DF,
    )


@app.cell
def _(
    DissolveSIMD_2012DF,
    DissolveSIMD_2016DF,
    DissolveSIMD_2020DF,
    cx,
    plt,
    simd_2012DF,
    simd_2016DF,
    simd_2020DF,
):
    fig, axes = plt.subplots(2, 3, figsize=(20, 20), sharex=True, sharey=True)
    ax = axes.flatten()

    plot_list = [
        simd_2012DF,
        simd_2016DF,
        simd_2020DF,
        DissolveSIMD_2012DF,
        DissolveSIMD_2016DF,
        DissolveSIMD_2020DF,
    ]
    proxydata_list = [
        "Income_Domain_2012_Rank",
        "Income_Domain_2016_Rank",
        "Income_Domain_2020_Rank",
        "weighted_Income_Domain_2012_Rank_decile",
        "weighted_Income_Domain_2016_Rank_decile",
        "weighted_Income_Domain_2020_Rank_decile",
    ]
    subtitles = [
        "SIMD Income 2012",
        "SIMD Income 2016",
        "SIMD Income 2020",
        "Dissolve Income 2012 (Weighted Decile)",
        "Dissolve Income 2016 (Weighted Decile)",
        "Dissolve Income 2020 (Weighted Decile)",
    ]

    for idx, (df, proxydata, titles) in enumerate(
        zip(plot_list, proxydata_list, subtitles)
    ):
        df.plot(proxydata, ax=ax[idx], legend=True, cmap="Spectral", alpha=0.7)
        ax[idx].set_title(titles)

        cx.add_basemap(ax[idx], crs=df.crs, source="OpenStreetMap Mapnik")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
