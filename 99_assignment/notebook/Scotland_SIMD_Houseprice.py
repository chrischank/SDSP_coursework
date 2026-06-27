import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore and Predict the Spatio-Temporal relationship of Scotland based on SIMD and Housing prices
    """)
    return


@app.cell
def _():
    ###############################
    # Notebook for SDSP Assessment#
    # Maintainer: Christopher Chan#
    # Version: 0.1.4              #
    # Date: 2026-06-27            #
    ###############################

    import re
    import sys
    import esda
    import itertools
    import statsmodels as sm
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    import contextily as cx
    import seaborn as sns
    import statsmodels.formula.api as smf
    from statsmodels.tools.tools import pinv_extended
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    from libpysal import graph
    from typing import Literal, Optional

    sys.path.append("..")
    from src.dictionary import (
        COUNCIL_ALIGNMENT,
        SIMD_DOMAIN_2012,
        SIMD_DOMAIN_2016,
        SIMD_DOMAIN_2020,
        INFLATION_2011,
    )

    DATA_RAW = Path("../data/01_raw")
    Path("../data/02_intermediate").mkdir(parents=True, exist_ok=True)
    Path("../data/03_feature").mkdir(parents=True, exist_ok=True)
    Path("../figure").mkdir(parents=True, exist_ok=True)
    DATA_INTERMEDIATE = Path("../data/02_intermediate")
    DATA_FEATURE = Path("../data/03_feature")
    FIGURE = Path("../figure")
    return (
        COUNCIL_ALIGNMENT,
        DATA_FEATURE,
        DATA_INTERMEDIATE,
        DATA_RAW,
        FIGURE,
        INFLATION_2011,
        Literal,
        Optional,
        Path,
        SIMD_DOMAIN_2012,
        SIMD_DOMAIN_2016,
        SIMD_DOMAIN_2020,
        cx,
        esda,
        gpd,
        graph,
        itertools,
        mean_squared_error,
        mpatches,
        mticker,
        np,
        pd,
        pinv_extended,
        plt,
        re,
        sm,
        smf,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Data Preprocesseing
    #### SIMD
    1. Roll up SIMD to council area using volume weighted average
    2. Calculate rank -> Decile
    3. Calculate Diffs

    #### ROS
    1. Adjust median house prices for mortgage sheet to 2011 base price (inflation adjustment)
    2. Compute volume weighted median and mean for corresponding temporal alignment with SIMD
    3. Calculate Diffs

    ### Stage 1: SIMD Preprocessing
    """)
    return


@app.cell
def _(
    COUNCIL_ALIGNMENT,
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
        """Load, align and volume-weight a pair of SIMD years to council area.

        Reads the Data Zone geometry/indicator files for an earlier and later
        SIMD release, canonicalises the council names, dissolves each year up to
        council area with a polygon-count weighting and assigns weighted deciles.

        Parameters
        ----------
        simd_geom_past : Path
            Shapefile of Data Zone geometries for the earlier SIMD year.
        simd_data_past : Path
            CSV of SIMD indicators for the earlier year.
        simd_geom_future : Path
            Shapefile of Data Zone geometries for the later SIMD year.
        simd_data_future : Path
            CSV of SIMD indicators for the later year.
        diff : DIFF_PERIOD
            Which year pair to process, either ``"12-16"`` or ``"16-20"``.

        Returns
        -------
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
            The past and future Data Zone frames followed by their council-area
            dissolved, weighted-decile frames.
        """

        def _agg_dict(columns: list) -> dict:
            """Build a ``{column: "sum"}`` aggregation map for a dissolve.

            Parameters
            ----------
            columns : list
                Column names to sum during the dissolve.

            Returns
            -------
            dict
                Mapping of each column name to the ``"sum"`` aggregation.
            """
            agg_dict = {f"{col}": "sum" for col in columns}

            return agg_dict

        def _weighted_domain_dissolve(
            gdf: gpd.GeoDataFrame, dissolveby: str, domain_cols: list
        ) -> gpd.GeoDataFrame:
            """Dissolve to a key and divide each domain by its polygon count.

            Parameters
            ----------
            gdf : gpd.GeoDataFrame
                Data Zone level frame to dissolve.
            dissolveby : str
                Column to dissolve on (the council area key).
            domain_cols : list
                SIMD domain columns to sum and then weight.

            Returns
            -------
            gpd.GeoDataFrame
                Dissolved frame with a ``weighted_<col>`` column per domain.
            """
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
                thousands=",",
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

            # Each year spells council areas differently (underscores, "&" vs
            # "and", two renames). Canonicalise here so the dissolve key - and
            # every downstream cross-year merge - aligns all 32 areas.
            simd_pastDF["Council_Area"] = simd_pastDF["Council_Area"].map(COUNCIL_ALIGNMENT)
            simd_futureDF["Council_Area"] = simd_futureDF["Council_Area"].map(
                COUNCIL_ALIGNMENT
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
                simd_data_past,
                usecols=[*SIMD_DOMAIN_2016.keys()],
                thousands=",",
            )

            simd_geom_futureDF = gpd.read_file(simd_geom_future)
            # Check projection
            if simd_geom_pastDF.crs != 27700:
                simd_geom_pastDF = simd_geom_pastDF.to_crs(27700)
            if simd_geom_futureDF.crs != 27700:
                simd_geom_futureDF = simd_geom_futureDF.to_crs(27700)

            simd_data_futureDF = pd.read_csv(
                simd_data_future,
                usecols=[*SIMD_DOMAIN_2020.keys()],
                thousands=",",
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

            # Each year spells council areas differently (underscores, "&" vs
            # "and", two renames). Canonicalise here so the dissolve key - and
            # every downstream cross-year merge - aligns all 32 areas.
            simd_pastDF["Council_Area"] = simd_pastDF["Council_Area"].map(COUNCIL_ALIGNMENT)
            simd_futureDF["Council_Area"] = simd_futureDF["Council_Area"].map(
                COUNCIL_ALIGNMENT
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

    return DIFF_PERIOD, simd_preprocessing


@app.cell
def _(DATA_INTERMEDIATE, DATA_RAW, Path, gpd, simd_preprocessing):
    if not Path(f"{DATA_INTERMEDIATE}/DissolveSIMD_2020.geojson").exists():
        print("Running SIMD preprocessing")

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
        simd_2020DF.to_file(f"{DATA_INTERMEDIATE}/simd_2020.geojson", driver="GeoJSON")

        DissolveSIMD_2012DF.to_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2012.geojson", driver="GeoJSON"
        )
        DissolveSIMD_2016DF.to_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2016.geojson", driver="GeoJSON"
        )
        DissolveSIMD_2020DF.to_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2020.geojson", driver="GeoJSON"
        )

    else:
        print("SIMD data already exists - skip preprocessing and read from intermediate")
        simd_2012DF = gpd.read_file(f"{DATA_INTERMEDIATE}/simd_2012.geojson")
        simd_2016DF = gpd.read_file(f"{DATA_INTERMEDIATE}/simd_2016.geojson")
        simd_2020DF = gpd.read_file(f"{DATA_INTERMEDIATE}/simd_2020.geojson")

        DissolveSIMD_2012DF = gpd.read_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2012.geojson"
        )
        DissolveSIMD_2016DF = gpd.read_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2016.geojson"
        )
        DissolveSIMD_2020DF = gpd.read_file(
            f"{DATA_INTERMEDIATE}/DissolveSIMD_2020.geojson"
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
def _():
    # fig, axes = plt.subplots(2, 3, figsize=(20, 20), sharex=True, sharey=True)
    # ax = axes.flatten()
    #
    # plot_list = [
    #    simd_2012DF,
    #    simd_2016DF,
    #    simd_2020DF,
    #    DissolveSIMD_2012DF,
    #    DissolveSIMD_2016DF,
    #    DissolveSIMD_2020DF,
    # ]
    # proxydata_list = [
    #    "Income_Domain_2012_Rank",
    #    "Income_Domain_2016_Rank",
    #    "Income_Domain_2020_Rank",
    #    "weighted_Income_Domain_2012_Rank_decile",
    #    "weighted_Income_Domain_2016_Rank_decile",
    #    "weighted_Income_Domain_2020_Rank_decile",
    # ]
    # subtitles = [
    #    "SIMD Income 2012",
    #    "SIMD Income 2016",
    #    "SIMD Income 2020",
    #    "Dissolve Income 2012 (Weighted Decile)",
    #    "Dissolve Income 2016 (Weighted Decile)",
    #    "Dissolve Income 2020 (Weighted Decile)",
    # ]
    #
    # for idx, (df, proxydata, titles) in enumerate(
    #    zip(plot_list, proxydata_list, subtitles)
    # ):
    #    df.plot(proxydata, ax=ax[idx], legend=True, cmap="Spectral", alpha=0.7)
    #    ax[idx].set_title(titles)
    #
    #    cx.add_basemap(ax[idx], crs=df.crs, source="CartoDB DarkMatter")
    #
    # plt.tight_layout()
    # fig.savefig(FIGURE / "simd_income_maps.png", dpi=150, bbox_inches="tight")
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Let's explore the randomness in both SIMD original and weighted quantiles

    While I can use queen cardinalities in higher resolution Data Zones, I cannot use them in lower resolution Council Areas due large areas often covering an entire Island without contiguity.

    In order to account for both tobler's law queen cardinality and the island effect, I can take 2 approaches:
    1. Queen union KNN
    or
    2. Hyperparameter optimised KNN.

    I will use the first approach for exploration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Global spatial autocorrelation for simd domains
    """)
    return


@app.cell
def _(
    DissolveSIMD_2012DF,
    DissolveSIMD_2016DF,
    DissolveSIMD_2020DF,
    SIMD_DOMAIN_2012,
    SIMD_DOMAIN_2016,
    SIMD_DOMAIN_2020,
    esda,
    np,
    plt,
    simd_2012DF,
    simd_2016DF,
    simd_2020DF,
):
    # First find the optimal k for original simd domain
    def plot_k_elbow(year: int, dissolve: bool, figure_dir: "Path") -> plt.Figure:
        """Plot Moran's I against K for each SIMD domain to find the elbow.

        Parameters
        ----------
        year : int
            SIMD year to plot (2012, 2016 or 2020).
        dissolve : bool
            If ``True`` use the council-area dissolved frame, else the Data Zone
            frame.
        figure_dir : Path
            Directory the correlogram figure is saved into.

        Returns
        -------
        plt.Figure
            The correlogram figure with one subplot per domain.
        """
        meta_cols = ["Data_Zone", "Council_Area", "Intermediate_Zone", "geometry"]

        domain_df_dict = {
            (2012, False): simd_2012DF,
            (2012, True): DissolveSIMD_2012DF,
            (2016, False): simd_2016DF,
            (2016, True): DissolveSIMD_2016DF,
            (2020, False): simd_2020DF,
            (2020, True): DissolveSIMD_2020DF,
        }

        col_dict = {
            2012: [*SIMD_DOMAIN_2012.values()],
            2016: [*SIMD_DOMAIN_2016.values()],
            2020: [*SIMD_DOMAIN_2020.values()],
        }

        domain_df = domain_df_dict[(year, dissolve)]

        if dissolve:
            domain_cols = [
                col
                for col in domain_df.columns
                if col.startswith("weighted")
                and col.endswith("_Rank")
                and col not in meta_cols
            ]
        else:
            domain_cols = [col for col in col_dict[year] if col not in meta_cols]

        print(f"Finding K for {domain_cols}")

        fig, axes = plt.subplots(4, 2, figsize=(20, 10))
        ax = axes.flatten()

        repr_points = domain_df.representative_point()

        if not dissolve:
            # k = np.arange(1, 1000, 5).tolist()
            k = [5, 10, 25, 50, 75, 100, 250, 500, 1000]
        else:
            k = np.arange(1, 32, 1)

        for idx, domain in enumerate(domain_cols):
            simd_correlogram = esda.correlogram(
                geometry=repr_points,
                variable=domain_df[domain],
                support=k,
                distance_type="knn",
            )

            simd_correlogram.I.plot(ax=ax[idx], marker="o")
            if dissolve:
                ax[idx].set_title(f"Dissolve {domain} - Year {year}")
            else:
                ax[idx].set_title(f"{domain} - Year {year}")
            ax[idx].set_xlabel("K-Nearest Neighbours")
            ax[idx].set_ylabel("Moran's I")

        for i in range(len(domain_cols), len(ax)):
            fig.delaxes(ax[i])

        plt.tight_layout()
        fig.savefig(
            figure_dir / f"k_elbow_{'dissolve' if dissolve else 'raw'}_{year}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

        return fig

    return (plot_k_elbow,)


@app.cell
def _(FIGURE, plot_k_elbow):
    k_2012 = plot_k_elbow(2012, dissolve=True, figure_dir=FIGURE)
    k_2016 = plot_k_elbow(2016, dissolve=True, figure_dir=FIGURE)
    k_2020 = plot_k_elbow(2020, dissolve=True, figure_dir=FIGURE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    KNN exploration of multiple decile suggests that spatial autocorrelation does not converge well with higher cluster, suggesting that I might need to include contiguity in combination with K = 3

    Particularly for factors other than Health, most have no spatial autocorrelation, suggesting that I might need to include contiguity in combination with K = 3
    To see if I can improve the Moran's I
    """)
    return


@app.cell
def _(
    DissolveSIMD_2012DF,
    DissolveSIMD_2016DF,
    DissolveSIMD_2020DF,
    Optional,
    SIMD_DOMAIN_2012,
    SIMD_DOMAIN_2016,
    SIMD_DOMAIN_2020,
    esda,
    gpd,
    graph,
    simd_2012DF,
    simd_2016DF,
    simd_2020DF,
):
    # Let's explore the queen contiguity
    def moran_local(
        year: Optional[int],
        dissolve: bool,
        diff: bool,
        diff_df: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """Attach local Moran's I cluster labels for each SIMD domain.

        Builds a combined queen-contiguity + KNN3 weights matrix and runs a LISA
        for the queen, KNN3 and combined graphs, writing the combined Moran's I,
        p-value and the per-graph cluster labels back onto the frame.

        Parameters
        ----------
        year : Optional[int]
            SIMD year to select when ``diff`` is ``False``; ignored otherwise.
        dissolve : bool
            If ``True`` operate on the council-area dissolved frame.
        diff : bool
            If ``True`` operate on a difference frame supplied via ``diff_df``.
        diff_df : Optional[gpd.GeoDataFrame], optional
            Difference frame used when ``diff`` is ``True``, by default ``None``.

        Returns
        -------
        gpd.GeoDataFrame
            The input frame with added Moran's I, p-value and cluster columns.
        """
        meta_cols = ["Data_Zone", "Council_Area", "Intermediate_Zone", "geometry"]

        domain_df_dict = {
            (2012, False, False): simd_2012DF,
            (2012, True, False): DissolveSIMD_2012DF,
            (2016, False, False): simd_2016DF,
            (2016, True, False): DissolveSIMD_2016DF,
            (2020, False, False): simd_2020DF,
            (2020, True, False): DissolveSIMD_2020DF,
        }

        col_dict = {
            2012: [*SIMD_DOMAIN_2012.values()],
            2016: [*SIMD_DOMAIN_2016.values()],
            2020: [*SIMD_DOMAIN_2020.values()],
        }

        if dissolve and not diff:
            domain_df = domain_df_dict[(year, dissolve, diff)]
            domain_cols = [
                col
                for col in domain_df.columns
                if col.startswith("weighted")
                and col.endswith("_Rank")
                and col not in meta_cols
            ]
        elif dissolve and diff:
            domain_df = diff_df
            domain_cols = [
                col
                for col in domain_df.columns
                if col.startswith("diff") and col.endswith("_Rank") and col not in meta_cols
            ]
        else:
            domain_df = domain_df_dict[(year, dissolve, diff)]
            domain_cols = [col for col in col_dict[year] if col not in meta_cols]

        contiguity_graph = graph.Graph.build_contiguity(domain_df, rook=False)
        knn3_graph = graph.Graph.build_knn(domain_df.centroid, k=3)

        contiguity_r = contiguity_graph.transform("r")
        knn3_r = knn3_graph.transform("r")
        combi_graph = graph.Graph.union(contiguity_graph, knn3_graph)
        combi_w = combi_graph.transform("r")

        for idx, col in enumerate(domain_cols):
            lisa_queen = esda.Moran_Local(domain_df[col], contiguity_r, permutations=999)
            lisa_knn3 = esda.Moran_Local(domain_df[col], knn3_r, permutations=999)
            lisa_combine = esda.Moran_Local(domain_df[col], combi_w, permutations=999)
            domain_df[f"combine_moran'sI_{col}"] = lisa_combine.Is
            domain_df[f"combine_pvalue_{col}"] = lisa_combine.p_sim
            # Due to spatial aggregation, replaxing the p-value
            domain_df[f"queen_cluster_{col}"] = lisa_queen.get_cluster_labels(
                crit_value=0.1
            )
            domain_df[f"knn3_cluster_{col}"] = lisa_knn3.get_cluster_labels(crit_value=0.1)
            domain_df[f"combine_cluster_{col}"] = lisa_combine.get_cluster_labels(
                crit_value=0.1
            )

        return domain_df

    return (moran_local,)


@app.cell
def _(moran_local):
    cluster_Dissolve2012 = moran_local(2012, dissolve=True, diff=False)
    cluster_Dissolve2016 = moran_local(2016, dissolve=True, diff=False)
    cluster_Dissolve2020 = moran_local(2020, dissolve=True, diff=False)
    return cluster_Dissolve2012, cluster_Dissolve2016, cluster_Dissolve2020


@app.cell
def _(Literal, cx, plt, sns):
    GRAPH_TYPE = Literal["knn3", "queen", "combine"]


    def plot_lisa(
        graph_df: "gpd.GeoDataFrame",
        method: GRAPH_TYPE,
        name: str,
        figure_dir: "Path",
    ) -> plt.Figure:
        """Map the LISA cluster labels for one weights method across domains.

        Parameters
        ----------
        graph_df : gpd.GeoDataFrame
            Frame carrying ``<method>_cluster_*`` columns from ``moran_local``.
        method : GRAPH_TYPE
            Weights graph to plot: ``"knn3"``, ``"queen"`` or ``"combine"``.
        name : str
            Label used in the saved figure file name.
        figure_dir : Path
            Directory the LISA figure is saved into.

        Returns
        -------
        plt.Figure
            Figure of cluster maps plus a stacked cluster-count histogram.
        """
        cluster_cols = [
            col for col in graph_df.columns if col.startswith(f"{method}_cluster")
        ]

        fig, axes = plt.subplots(2, 4, figsize=(30, 15))
        ax = axes.flatten()

        color_map = {
            "High-High": "#2c7bb6",
            "High-Low": "#abd9e9",
            "Low-High": "#fdae61",
            "Low-Low": "#d7191c",
            "Insignificant": "lightgrey",
        }

        for idx, col in enumerate(cluster_cols):
            # Map values to colors. Unmapped categories default to transparent/white.
            colors = graph_df[col].map(color_map).fillna("none")

            graph_df.plot(ax=ax[idx], color=colors, alpha=0.7)
            ax[idx].set_title(col)

            cx.add_basemap(ax[idx], crs=graph_df.crs, source="CartoDB DarkMatter")

        # Melt graph_df for histplot
        graph_df_long = graph_df.melt(
            value_vars=cluster_cols, var_name="Domain", value_name="Cluster Type"
        )
        sns.histplot(
            graph_df_long,
            y="Domain",
            hue="Cluster Type",
            hue_order=[*color_map.keys()],
            palette=color_map,
            multiple="stack",
            ax=ax[-1],
        )
        ax[-1].set_yticklabels(
            [
                "Health",
                "Income",
                "Employment",
                "Education",
                "Housing",
                "Geographic Access",
                "Crime",
            ]
        )

        plt.tight_layout()
        fig.savefig(figure_dir / f"lisa_{method}_{name}.png", dpi=150, bbox_inches="tight")
        return fig

    return (plot_lisa,)


@app.cell
def _(
    FIGURE,
    cluster_Dissolve2012,
    cluster_Dissolve2016,
    cluster_Dissolve2020,
    plot_lisa,
):
    combine2012_moran = plot_lisa(
        cluster_Dissolve2012, method="combine", name="dissolve_2012", figure_dir=FIGURE
    )
    combine2016_moran = plot_lisa(
        cluster_Dissolve2016, method="combine", name="dissolve_2016", figure_dir=FIGURE
    )
    combine2020_moran = plot_lisa(
        cluster_Dissolve2020, method="combine", name="dissolve_2020", figure_dir=FIGURE
    )
    return


@app.cell
def _(
    DATA_INTERMEDIATE,
    cluster_Dissolve2012,
    cluster_Dissolve2016,
    cluster_Dissolve2020,
    gpd,
    re,
):
    # Calculate the diff in volume weighted rank
    def calculate_weighted_diff(diff_year: int) -> gpd.GeoDataFrame:
        """Compute the future-minus-past change in weighted SIMD ranks.

        Parameters
        ----------
        diff_year : int
            Period encoded as ``1216`` (2012->2016) or ``1620`` (2016->2020).

        Returns
        -------
        gpd.GeoDataFrame
            Council-area frame with a ``diff_<domain>_Rank`` column per domain.
        """
        match diff_year:
            case 1216:
                past_df = cluster_Dissolve2012
                future_df = cluster_Dissolve2016
            case 1620:
                past_df = cluster_Dissolve2016
                future_df = cluster_Dissolve2020

        gdf_merge = past_df.merge(future_df, on="Council_Area", how="inner")
        if len(gdf_merge) != len(past_df):
            print(
                f"WARNING: diff merge kept {len(gdf_merge)}/{len(past_df)} "
                "council areas - name mismatch remains"
            )
        gdf_merge = gdf_merge.drop("geometry_y", axis=1).rename(
            columns={"geometry_x": "geometry"}
        )

        gdf_merge = gdf_merge[
            ["Council_Area", "geometry"]
            + [col for col in gdf_merge.columns if col.startswith("weighted_")]
        ]

        past_cols = [
            col
            for col in past_df.columns
            if col.startswith("weighted_") and not col.endswith("_decile")
        ]
        future_cols = [
            col
            for col in future_df.columns
            if col.startswith("weighted_") and not col.endswith("_decile")
        ]

        for colp, colf in zip(past_cols, future_cols):
            coln = (
                "diff_"
                + colp.replace("_Rank", "")
                + "_"
                + re.sub(r"\D", "", colf)
                + "_Rank"
            )
            gdf_merge[coln] = gdf_merge[colf] - gdf_merge[colp]

        gdf_merge.to_file(
            f"{DATA_INTERMEDIATE}/simd_diff_{diff_year}.geojson", driver="GeoJSON"
        )

        return gdf_merge

    return (calculate_weighted_diff,)


@app.cell
def _(DATA_INTERMEDIATE, Path, calculate_weighted_diff, gpd):
    if Path.exists(f"{DATA_INTERMEDIATE}/simd_diff_1620.geojson"):
        print("Diff simd dataframes already exist - skipping recalculation")
        SIMDdiff_1216 = gpd.read_file(f"{DATA_INTERMEDIATE}/simd_diff_1216.geojson")
        SIMDdiff_1620 = gpd.read_file(f"{DATA_INTERMEDIATE}/simd_diff_1620.geojson")
    else:
        print("Diff simd dataframes do not exist - calculating")
        SIMDdiff_1216 = calculate_weighted_diff(diff_year=1216)
        SIMDdiff_1620 = calculate_weighted_diff(diff_year=1620)
    return SIMDdiff_1216, SIMDdiff_1620


@app.cell
def _(
    FIGURE,
    SIMDdiff_1216,
    SIMDdiff_1620,
    cluster_Dissolve2012,
    cluster_Dissolve2016,
    cluster_Dissolve2020,
    pd,
    plt,
    sns,
):
    # Lets plot the absolute changes and the diffs side by side
    simd_mergeDF = pd.merge(
        cluster_Dissolve2012, cluster_Dissolve2016, on="Council_Area"
    ).merge(cluster_Dissolve2020, on="Council_Area")
    simd_mergeDF = simd_mergeDF[
        [
            col
            for col in simd_mergeDF.columns
            if col.startswith("weighted_") or col in ["Council_Area"]
        ]
    ]
    simd_diffDF = pd.merge(SIMDdiff_1216, SIMDdiff_1620, on="Council_Area")
    simd_diffDF = simd_diffDF[
        [
            col
            for col in simd_diffDF.columns
            if col.startswith("diff_") or col in ["Council_Area"]
        ]
    ]

    simd_mergeDFlong = pd.melt(
        simd_mergeDF, id_vars=["Council_Area"], var_name="simd", value_name="value"
    )
    simd_mergeDFlong["year"] = simd_mergeDFlong["simd"].str.extract(r"(\d{4})")
    simd_mergeDFlong["domain"] = simd_mergeDFlong["simd"].str.split("_").str[1]

    simd_diffDFlong = pd.melt(
        simd_diffDF, id_vars=["Council_Area"], var_name="simd", value_name="value"
    )
    simd_diffDFlong["year"] = (
        simd_diffDFlong["simd"]
        .str.extract(r"(\d{4}_\d{4})", expand=False)
        .str.replace("_", "-")
    )
    simd_diffDFlong["domain"] = simd_diffDFlong["simd"].str.split("_").str[2]


    def plot_abs_diff(
        simd_mergeDFlong: pd.DataFrame, simd_diffDFlong: pd.DataFrame
    ) -> None:
        """Plot absolute and difference SIMD rank trajectories side by side.

        Parameters
        ----------
        simd_mergeDFlong : pd.DataFrame
            Long-format absolute weighted ranks with ``year`` and ``domain``.
        simd_diffDFlong : pd.DataFrame
            Long-format per-period rank differences with ``year`` and ``domain``.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        ax = axes.flatten()

        palette = sns.color_palette("gist_ncar", 32)
        max_abs = simd_mergeDFlong["value"].abs().max()
        max_diff = simd_diffDFlong["value"].abs().max()

        for idx, domain in enumerate(simd_mergeDFlong["domain"].unique()):
            ax_abs, ax_diff = ax[idx * 2], ax[idx * 2 + 1]

            subset_abs = simd_mergeDFlong[simd_mergeDFlong["domain"] == domain]
            subset_diff = simd_diffDFlong[simd_diffDFlong["domain"] == domain]

            sns.lineplot(
                subset_abs,
                x="year",
                y="value",
                hue="Council_Area",
                palette=palette,
                ax=ax_abs,
                legend=(idx == 0),
                errorbar=None,
            )

            sns.lineplot(
                subset_diff,
                x="year",
                y="value",
                hue="Council_Area",
                palette=palette,
                ax=ax_diff,
                legend=False,
                errorbar=None,
            )

            ax_abs.set_ylim(0, max_abs)

            ax_diff.set_ylim(-max_diff, max_diff)
            ax_diff.axhline(0, color="black", linestyle="--")

            ax_abs.set_title(domain + " (absolute)")
            ax_diff.set_title(domain + " (difference)")

        for i in range(len(simd_mergeDFlong["domain"].unique()) * 2, len(ax)):
            fig.delaxes(ax[i])

        ax[0].legend_.remove()

        handles, labels = ax[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.07),
            ncol=4,
            title="Council Area",
            fontsize="small",
        )

        plt.tight_layout()
        fig.savefig(FIGURE / "simd_abs_vs_diff.png", dpi=150, bbox_inches="tight")
        plt.show()


    plot_abs_diff(simd_mergeDFlong, simd_diffDFlong)
    return


@app.cell
def _(FIGURE, SIMDdiff_1216, SIMDdiff_1620, moran_local, plot_lisa):
    diffcluster_SIMD1216 = moran_local(
        year=None, dissolve=True, diff=True, diff_df=SIMDdiff_1216
    )
    diffcluster_SIMD1620 = moran_local(
        year=None, dissolve=True, diff=True, diff_df=SIMDdiff_1620
    )
    combinediff_1216moran = plot_lisa(
        diffcluster_SIMD1216, method="combine", name="diff_1216", figure_dir=FIGURE
    )
    combinediff_1620moran = plot_lisa(
        diffcluster_SIMD1620, method="combine", name="diff_1620", figure_dir=FIGURE
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 1: SIMD preprocessing summary

    I have rolled-up the SIMD data into Council Areas, using a volume weighted approach for each year. I calculated the Moran's I for each domain using a lower critical level of 0.1.

    #### 1. Absolute numbers
    1. For all periods, the Health, Income, and Employment consistently has the highest proportion of councils with statistical significance clustering.
    2. The Councils of Highlands, Shetlands, and Orkney Islands have experienced consistent clustering for Health, Income, Crime and Employment domains, being above average and significance in their neighbouring effects. This suggests that for many of the northern councils, their spatial relations are much more pronounced and less random than the rest of Scotland. Unsurprisingly since the Highlands and Islands have always been closely intertwined economically and culturally.
    3. For the council of South Ayrshire and East Dunbartonshire in Western Central Belt, they experienced consistently Low-Low clustering for Health, Income, and Employment.

    #### 2. Relative differences
    1. This is a significantly more interesting metric. Looking at the relative lineplots, we saw a smaller overall increase of ranking for Health, Education and Geographic Access for 2016-2020 when compared to 2012-2016, while many other domains experienced a reduction in improvement for most councils. Looking at cluster for diff for Highlands, Shetlands, and Orkney Islands, although absolute rankings suggests they are consistently above average for Income and Employment, relative differences suggest that high absolute spatial relationship do not translate to relative. Particularly for Income and Employment, the Shetland and Orkney Islands experiencing negative changes with significant negative changes in the surrounding areas. I.e. quality of life for Income and Employment domain are high but worsening.
    2. Meanwhile for Income, Employment, and Crime 2012-2016, relative differences in Low-Low clustering have gain grounds in the borders of Scotland, suggesting lower than average changes begets lower changes in these domains in the surround areas. But this was not significant in absolute ranking for 2012 and 2016. This might suggest that the border regions are experiencing increase in crime, and negative income and employment outcome caused by neighbouring effects temporarily.
    3. The largest change came from Income, Employment, and Crime. We saw a reversal in changes trend particularly pronounced for employment where Dumfries and Galloway had lower than average changes to higher than average changes.

    **Overall:** The result might indicate that the more affluent rural areas in the north of Scotland with strong autocorrelation might be experience negative changes recently. While more economically deprived areas of Western Scotland (Ayrshire) have not experienced much changes at all, with the rest of the central belt being a mixed bag results, with central Scotland experiencing improvement in Income and Employment for 2016-2020. The borders and Dumfries and Galloway are also inconsistent. Excluding spatial autocorrelation, we generally saw stagnation with 2016-2020 saw lower positive changes in improvement or worsening compared to 2012-2016. Growth in quality of life have generally reduced, with Crime seeing mixed result. Most significant was housing where there were little to no changes in ranking for 2016-2020. Employment and Income were both trending into the negative with 2016-2020, suggesting that income and employment metrics have worsen overall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 2: House purchase price preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I will be using the ROS house median house price dataset sheet C6: Median, mean, volume and value of all residential market value property sales by funding status and local authority: Scotland, since 2004, calendar year data
    """)
    return


@app.cell
def _(DATA_RAW, pd):
    ros_salesDF = pd.read_excel(
        f"{DATA_RAW}/ros_all_stats_March_2026.xlsx",
        sheet_name="C6",
        skiprows=5,
        usecols=range(1, 8),
    )
    ros_salesDF["Funding status"] = ros_salesDF["Funding status"].str.replace(
        r"^\d{1}\s-\s", "", regex=True
    )
    ros_salesDF = ros_salesDF[ros_salesDF["Local authority"] != "Scotland"]

    # Council name alignment for Na h-Eileanan an Iar
    ros_salesDF["Local authority"] = ros_salesDF["Local authority"].str.replace(
        "Na h-Eileanan Siar", "Na h-Eileanan an Iar", regex=False
    )
    return (ros_salesDF,)


@app.cell
def _(INFLATION_2011, itertools, mticker, pd, plt, sns):
    def plot_ros_scatter(gdf: pd.DataFrame, figure_dir: "Path") -> None:
        """Plot inflation-adjusted ROS price trends with SIMD-year markers.

        Parameters
        ----------
        gdf : pd.DataFrame
            ROS sales table with calendar year, council, funding status and
            mean/median prices.
        figure_dir : Path
            Directory the scatter-grid figure is saved into.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """
        # Apply inflation adjustment
        inflation = gdf["Calendar year"].map(INFLATION_2011)
        gdf_copy = gdf.copy()
        gdf_copy["Mean residential property price (£)"] *= inflation
        gdf_copy["Median residential property price (£)"] *= inflation

        melted_gdf = pd.melt(
            gdf_copy,
            id_vars=["Calendar year", "Local authority", "Funding status"],
            value_vars=[
                "Median residential property price (£)",
                "Mean residential property price (£)",
            ],
            var_name="Price Stat",
            value_name="Adjusted Price (£)",
        )

        melted_gdf["Price Stat"] = melted_gdf["Price Stat"].str.replace(
            " residential property price (£)", ""
        )

        n = gdf["Local authority"].nunique()
        markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "P", "X", "d"]

        mlist = list(itertools.islice(itertools.cycle(markers), n))

        g = sns.lmplot(
            data=melted_gdf,
            x="Calendar year",
            y="Adjusted Price (£)",
            hue="Local authority",
            col="Funding status",
            row="Price Stat",
            markers=mlist,
            ci=None,
            height=5,
            aspect=1.2,
            legend=True,
            fit_reg=True,
            order=3,
            facet_kws={"margin_titles": True},
        )

        simd_years = [2012, 2016, 2020]
        linestyles = [":", "--", "-."]

        for ax in g.axes.flat:
            for year, style in zip(simd_years, linestyles):
                ax.axvline(x=year, color="black", linestyle=style, zorder=0)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        g.set_axis_labels("Calendar year", "Adjusted Price (£)")

        # Save and display
        g.fig.savefig(figure_dir / "ros_scatter_grid.png", dpi=150, bbox_inches="tight")
        plt.show()

    return (plot_ros_scatter,)


@app.cell
def _(FIGURE, plot_ros_scatter, ros_salesDF):
    plot_ros_scatter(ros_salesDF, FIGURE)
    return


@app.cell
def _(INFLATION_2011, gpd, pd):
    def ros_volume_weighted(
        df: pd.DataFrame, simd_year: int, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Volume-weight ROS prices over the four years before a SIMD year.

        Filters sales to the ``simd_year - 4`` to ``simd_year`` window, applies
        the 2011-base inflation adjustment and computes a volume-weighted mean
        and median price per council area and funding status, then joins the
        council geometry.

        Parameters
        ----------
        df : pd.DataFrame
            Raw ROS sales table (sheet C6).
        simd_year : int
            SIMD year the price window is aligned to (2012, 2016 or 2020).
        gdf : gpd.GeoDataFrame
            Council-area frame providing geometry to join on.

        Returns
        -------
        gpd.GeoDataFrame
            Volume-weighted prices per council area, funding status and year
            range, with geometry attached.
        """

        # 1. Standardize columns
        df = df.rename(
            columns={
                "Calendar year": "Year",
                "Local authority": "Council_Area",
                "Local authority code": "Council_Code",
                "Funding status": "Funding_Status",
                "Median residential property price (£)": "Median_Price",
                "Mean residential property price (£)": "Mean_Price",
                "Volume of residential property sales": "Volume",
            }
        )

        mask = df["Year"].between(simd_year - 4, simd_year, inclusive="left")
        mask_df = df.loc[mask].copy()

        # Baseline inflation adjustment
        inflation = mask_df["Year"].map(INFLATION_2011)
        mask_df["Mean_Price"] *= inflation
        mask_df["Median_Price"] *= inflation

        match simd_year:
            case 2012:
                mask_df["Year_Range"] = "2008-2011"
            case 2016:
                mask_df["Year_Range"] = "2012-2015"
            case 2020:
                mask_df["Year_Range"] = "2016-2019"

        def _calculate_weighted_metrics(group: pd.DataFrame) -> pd.Series:
            """Volume-weight the mean and median price within one group.

            Parameters
            ----------
            group : pd.DataFrame
                Sales rows for a single council/funding-status/year-range group.

            Returns
            -------
            pd.Series
                The volume-weighted mean, weighted median and total volume.
            """
            w = group["Volume"]
            w_sum = w.sum()

            vw_mean = (group["Mean_Price"] * w).sum() / w_sum

            g_sorted = group.dropna(subset=["Median_Price", "Volume"]).sort_values(
                "Median_Price"
            )
            cumsum = g_sorted["Volume"].cumsum()
            vw_median = g_sorted.loc[cumsum >= (w_sum / 2.0), "Median_Price"].iloc[0]

            return pd.Series(
                {
                    "VW_Mean_Price": vw_mean,
                    "VW_Median_Price": vw_median,
                    "Total_Volume": w_sum,
                }
            )

        agg_df = (
            mask_df.groupby(
                ["Year_Range", "Council_Area", "Council_Code", "Funding_Status"]
            )
            .apply(_calculate_weighted_metrics, include_groups=False)
            .reset_index()
            .drop_duplicates()
        )

        merged_df = pd.merge(
            agg_df, gdf[["Council_Area", "geometry"]], on="Council_Area", how="left"
        )

        return gpd.GeoDataFrame(merged_df, geometry="geometry", crs=gdf.crs)

    return (ros_volume_weighted,)


@app.cell
def _(cluster_Dissolve2012, ros_salesDF, ros_volume_weighted):
    ros_simd2012_df = ros_volume_weighted(ros_salesDF, 2012, cluster_Dissolve2012)
    ros_simd2016_df = ros_volume_weighted(ros_salesDF, 2016, cluster_Dissolve2012)
    ros_simd2020_df = ros_volume_weighted(ros_salesDF, 2020, cluster_Dissolve2012)
    return ros_simd2012_df, ros_simd2016_df, ros_simd2020_df


@app.cell
def _(gpd, pd, ros_simd2012_df, ros_simd2016_df, ros_simd2020_df):
    ros_concat_df = gpd.GeoDataFrame(
        pd.concat(
            [ros_simd2012_df, ros_simd2016_df, ros_simd2020_df],
            axis=0,
            ignore_index=True,
        )
    ).reset_index()
    return (ros_concat_df,)


@app.cell
def _(cx, np, plt):
    # Lets plot and explore the volume weighted average price for different funding status
    def plot_house_price(
        vw_gdf: "gpd.GeoDataFrame", average: str, figure_dir: "Path"
    ) -> None:
        """Map a volume-weighted price column across funding status and period.

        Parameters
        ----------
        vw_gdf : gpd.GeoDataFrame
            Volume-weighted price frame from ``ros_volume_weighted``.
        average : str
            Price column to map, ``"VW_Mean_Price"`` or ``"VW_Median_Price"``.
        figure_dir : Path
            Directory the price map figure is saved into.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        ax = axes.flatten()
        idx = 0

        min_price = np.nanmin(vw_gdf[["VW_Mean_Price", "VW_Median_Price"]].to_numpy())
        max_price = np.nanmax(vw_gdf[["VW_Mean_Price", "VW_Median_Price"]].to_numpy())

        for status in vw_gdf["Funding_Status"].unique():
            for year in vw_gdf["Year_Range"].unique():
                filter_gdf = vw_gdf[
                    (vw_gdf["Year_Range"] == year) & (vw_gdf["Funding_Status"] == status)
                ]
                filter_gdf.plot(
                    average,
                    ax=ax[idx],
                    cmap="RdYlGn_r",
                    vmin=min_price,
                    vmax=max_price,
                    legend=True,
                    alpha=0.7,
                )

                cx.add_basemap(ax[idx], crs=filter_gdf.crs, source="CartoDB DarkMatter")

                ax[idx].set_title(f"{year} {status} - ({average})")

                idx += 1

        plt.tight_layout()
        fig.savefig(figure_dir / f"house_price_{average}.png", dpi=150, bbox_inches="tight")
        plt.show()

    return (plot_house_price,)


@app.cell
def _(FIGURE, plot_house_price, ros_concat_df):
    plot_house_price(ros_concat_df, "VW_Median_Price", figure_dir=FIGURE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 3: Modelling
    I will perform 4 linear regression models:
    1. OLS with absolute SIMD and ROS Funding Status
    2. Spatial Regression with absolute SIMD and ROS Funding Status
    3. OLS with relative SIMD and ROS Funding Status
    4. Spatial Regression with relative SIMD and ROS Funding Status

    There are still some transformations that needed to be done before modelling:
    - One hot encoding of Council_Areas and Funding_Status
    - Preparing the differences for SIMD and ROS variables for relative models
    - Calculate relevant volume weighted mean and median prices and cluster on the volume weighted mean of Target, Predicted, and Residuals of the model outputs

    All models are fitted with elastic-net regularisation and tuned to minimise the MSE on a 20% validation set.

    #### OLS modelling
    """)
    return


@app.cell
def _(DATA_FEATURE, gpd, pd, re):
    def ols_preprocessing(
        df_2012: gpd.GeoDataFrame,
        df_2016: gpd.GeoDataFrame,
        df_2020: gpd.GeoDataFrame,
        ros_df: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Stack the three SIMD years and join the ROS prices for modelling.

        Trims each dissolved SIMD year to its first eight columns, strips the
        year from the domain names, concatenates the years and merges the
        volume-weighted ROS prices on council area and year range.

        Parameters
        ----------
        df_2012 : gpd.GeoDataFrame
            Dissolved SIMD 2012 frame.
        df_2016 : gpd.GeoDataFrame
            Dissolved SIMD 2016 frame.
        df_2020 : gpd.GeoDataFrame
            Dissolved SIMD 2020 frame.
        ros_df : gpd.GeoDataFrame
            Concatenated volume-weighted ROS price frame.

        Returns
        -------
        gpd.GeoDataFrame
            The joined SIMD/ROS modelling frame, also written to disk.
        """
        simd_df_list = [df_2012, df_2016, df_2020]

        # Assign new columns directly
        df_2012["Year_Range"] = "2008-2011"
        df_2016["Year_Range"] = "2012-2015"
        df_2020["Year_Range"] = "2016-2019"

        processed_df = []

        for df in simd_df_list:
            # FIX 1: Use .tolist() to flatten the Index into standard strings
            cols_to_keep = df.columns[:8].tolist() + ["Year_Range"]
            df_copy = df[cols_to_keep].copy()

            renamed_cols = [re.sub(r"\d{4}_", "", col) for col in df_copy.columns[1:8]]
            df_copy.columns = ["Council_Area"] + renamed_cols + ["Year_Range"]

            processed_df.append(df_copy)

        simd_concat_df = pd.concat(processed_df, axis=0, ignore_index=True)

        simd_ros_df = pd.merge(
            simd_concat_df, ros_df, on=["Council_Area", "Year_Range"], how="left"
        )
        simd_ros_df.drop(columns=["geometry_x"], inplace=True)
        simd_ros_df.rename(columns={"geometry_y": "geometry"}, inplace=True)

        # Dropping Funding Status == All Properties
        simd_ros_df.drop(
            simd_ros_df[simd_ros_df["Funding_Status"] == "All properties"].index,
            inplace=True,
        )

        simd_ros_gdf = gpd.GeoDataFrame(
            simd_ros_df, geometry="geometry", crs=simd_ros_df.crs
        )
        simd_ros_gdf.to_file(f"{DATA_FEATURE}/simd_ros.geojson", driver="GeoJSON")

        return simd_ros_gdf

    return (ols_preprocessing,)


@app.cell
def _(
    DATA_FEATURE,
    DissolveSIMD_2012DF,
    DissolveSIMD_2016DF,
    DissolveSIMD_2020DF,
    Path,
    gpd,
    ols_preprocessing,
    ros_concat_df,
):
    if not Path.exists(f"{DATA_FEATURE}/simd_ros.geojson"):
        print("SIMD ROS joined GeoJSON not found -- creating")
        simd_concat_df = ols_preprocessing(
            DissolveSIMD_2012DF, DissolveSIMD_2016DF, DissolveSIMD_2020DF, ros_concat_df
        )
    else:
        print("SIMD ROS joined GeoJSON found -- loading")
        simd_concat_df = gpd.read_file(f"{DATA_FEATURE}/simd_ros.geojson")
    return (simd_concat_df,)


@app.cell
def _(pd, simd_concat_df):
    # One-hot encoding of cateogorical variables
    simd_concat_OHdf = pd.get_dummies(
        simd_concat_df,
        columns=["Year_Range", "Funding_Status", "Council_Area"],
    ).drop(columns=["index"])
    return (simd_concat_OHdf,)


@app.function
def generate_xvar(df: "pd.DataFrame", price_prefix: str = "VW") -> list[str]:
    """Build the patsy predictor list, dropping identifiers and the response.

    Excludes identifier/geometry columns and any price response columns (the raw
    ``<prefix>_*`` set and the ``diff_<prefix>_*`` set) so they cannot leak in as
    predictors, then wraps each remaining column in ``Q('...')`` for patsy.

    Parameters
    ----------
    df : pd.DataFrame
        One-hot encoded modelling frame.
    price_prefix : str, optional
        Prefix identifying the price response columns, by default ``"VW"``.

    Returns
    -------
    list[str]
        Patsy-quoted predictor column names.
    """
    drop_cols = [col for col in ["Council_Code", "geometry"] if col in df.columns]

    price_cols = [
        col
        for col in df.columns
        if col.startswith(price_prefix) or col.startswith(f"diff_{price_prefix}")
    ]

    ros_X_var = df.columns.drop(drop_cols + price_cols)
    ros_X_var = [f"Q('{col}')" for col in ros_X_var]

    return ros_X_var


@app.cell
def _(
    itertools,
    mean_squared_error,
    np,
    pinv_extended,
    re,
    sm,
    smf,
    train_test_split,
):
    def _get_best_hyperparameters(
        formula: str,
        df: "pd.DataFrame",
        target: str,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Grid-search the elastic-net alpha / L1 ratio that minimises held-out MSE.

        Strips the patsy ``Q('...')`` wrappers to subset the frame, splits off a
        20% validation set and sweeps ``alpha`` (log-spaced) against the L1 weight,
        refitting a regularised OLS for each pair and keeping the lowest-MSE pair.

        Parameters
        ----------
        formula : str
            Patsy formula, ``"<target> ~ <predictors>"``.
        df : pd.DataFrame
            One-hot encoded modelling frame.
        target : str
            Response column name.
        random_state : int, optional
            Seed for the train/validation split, by default ``42``.

        Returns
        -------
        dict[str, float]
            ``{"alpha": ..., "l1_wt": ...}`` minimising the validation MSE.
        """
        cols_to_keep = [target] + generate_xvar(df)

        # Remove Patsy before subset and optimisation
        pattern = r"^Q\('(.*)'\)$"
        clean_cols = [re.sub(pattern, r"\1", col) for col in cols_to_keep]
        df_clean = df[clean_cols]

        train, val = train_test_split(df_clean, test_size=0.2, random_state=random_state)

        # Define the search space
        alphas = np.logspace(-4, 1, 20)
        l1_weights = np.arange(0, 1.1, 0.1)
        best_params = {"alpha": None, "l1_wt": None}
        min_mse = float("inf")

        test_model = smf.ols(formula, data=train)

        for alpha, l1_wt in itertools.product(alphas, l1_weights):
            try:
                res = test_model.fit_regularized(
                    method="elastic_net",
                    alpha=alpha,
                    L1_wt=l1_wt,
                    maxiter=100,
                )

                preds = res.predict(val)
                val_mse = mean_squared_error(val[target], preds)

                if val_mse < min_mse:
                    min_mse = val_mse
                    best_params = {"alpha": alpha, "l1_wt": l1_wt}

            except ValueError:
                continue

        print(
            f"Best parameters: alpha={best_params['alpha']:.5f}, l1_wt={best_params['l1_wt']} (MSE: {min_mse:.4f})"
        )
        return best_params


    def fit_regularized_ols(
        formula: str,
        df: "pd.DataFrame",
        target: str,
        random_state: int = 42,
    ) -> "OLSResults":
        """Fit an elastic-net OLS at its best hyperparameters as a summary-ready result.

        Grid-searches the regularisation strength with ``get_best_hyperparameters``,
        refits the elastic-net model on the full frame, then rebuilds an
        ``OLSResults`` from the regularised coefficients so the fit exposes
        ``.summary()``, ``.predict()`` and ``.resid`` like an ordinary OLS
        (``fit_regularized`` alone returns none of these). Used to fit all four
        linear models: absolute / relative (diff) x standard / spatial-lag.

        Parameters
        ----------
        formula : str
            Patsy formula, ``"<target> ~ <predictors>"``.
        df : pd.DataFrame
            One-hot encoded modelling frame.
        target : str
            Response column name.
        random_state : int, optional
            Seed forwarded to the hyperparameter search, by default ``42``.

        Returns
        -------
        OLSResults
            Regularised fit wrapped so inference output and prediction work.
        """
        best_params = _get_best_hyperparameters(
            formula, df, target, random_state=random_state
        )

        ols_model = smf.ols(formula, data=df)
        res = ols_model.fit_regularized(
            method="elastic_net",
            alpha=best_params["alpha"],
            L1_wt=best_params["l1_wt"],
        )

        # fit_regularized returns a RegularizedResults with no .summary()
        # rebuild OLSResults from the regularised params so the
        # standard inference table and .predict()/.resid become available.
        pinv_wexog, _ = pinv_extended(ols_model.wexog)
        norm_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

        return best_params, sm.regression.linear_model.OLSResults(
            ols_model, res.params, norm_cov_params
        )

    return (fit_regularized_ols,)


@app.cell
def _(fit_regularized_ols, simd_concat_OHdf):
    formula_simd_ros = f"VW_Median_Price ~ {' + '.join(generate_xvar(simd_concat_OHdf))}"
    ols_params, ols = fit_regularized_ols(
        formula_simd_ros, simd_concat_OHdf, "VW_Median_Price"
    )
    ols.summary()
    return ols, ols_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### OLS modelling (Spatial)
    """)
    return


@app.cell
def _(graph, pd):
    def calculate_combine_lag(
        concat_df: "gpd.GeoDataFrame",
        group_col: str = "Council_Code",
        period_col: "str | None" = None,
        xvar_prefix: str = "weighted_",
    ) -> "gpd.GeoDataFrame":
        """Add combined contiguity + KNN3 spatial lags of the predictor columns.

        Builds one combined weights matrix from the per-council geometries and,
        for each period, lags every ``xvar_prefix`` column so each council-period
        keeps its own neighbour-averaged value, then merges the lags back.

        Parameters
        ----------
        concat_df : gpd.GeoDataFrame
            Long modelling frame before one-hot encoding.
        group_col : str, optional
            Column identifying each area, by default ``"Council_Code"``.
        period_col : str | None, optional
            Column splitting the panel into periods; ``None`` lags the whole
            frame at once, by default ``None``.
        xvar_prefix : str, optional
            Prefix of the columns to lag, by default ``"weighted_"``.

        Returns
        -------
        gpd.GeoDataFrame
            ``concat_df`` with an added ``<col>_lag`` column per lagged predictor.
        """
        # Columns to calculate spatial lag: weighted_* (absolute) or diff_weighted_* (diff),
        # Funding_Status_* (absolute) or diff_Funding_Status_* (diff)
        xvar_cols = [col for col in concat_df.columns if col.startswith(xvar_prefix)]

        base_geoms = (
            concat_df.groupby(group_col).agg({"geometry": "first"}).set_geometry("geometry")
        )
        contiguity_graph = graph.Graph.build_contiguity(base_geoms)
        knn3_graph = graph.Graph.build_knn(base_geoms.centroid, k=3)
        combi_w = graph.Graph.union(contiguity_graph, knn3_graph).transform("r")

        lag_cols = [f"{col}_lag" for col in xvar_cols]

        periods = [None] if period_col is None else list(concat_df[period_col].unique())
        merge_keys = [group_col] if period_col is None else [group_col, period_col]

        lag_frames = []
        for period in periods:
            sub = (
                concat_df if period is None else concat_df[concat_df[period_col] == period]
            )
            agg_dict = {col: "first" for col in xvar_cols}
            agg_dict["geometry"] = "first"
            area_vals = (
                sub.groupby(group_col)
                .agg(agg_dict)
                .reindex(base_geoms.index)
                .set_geometry("geometry")
            )
            for col in xvar_cols:
                area_vals[f"{col}_lag"] = combi_w.lag(area_vals[col])

            frame = area_vals[lag_cols].reset_index()
            if period is not None:
                frame[period_col] = period
            lag_frames.append(frame)

        lag_df = pd.concat(lag_frames, ignore_index=True)
        spatial_concat_df = pd.merge(concat_df, lag_df, on=merge_keys, how="left")

        return spatial_concat_df

    return (calculate_combine_lag,)


@app.cell
def _(calculate_combine_lag, pd, simd_concat_df):
    # Lag on the pre-one-hot frame so the per-period (Year_Range) grouping can
    # give each year its own neighbour-averaged values, then one-hot encode.
    simd_spatial_df = calculate_combine_lag(simd_concat_df, period_col="Year_Range")
    simd_spatial_OHdf = pd.get_dummies(
        simd_spatial_df,
        columns=["Year_Range", "Funding_Status", "Council_Area"],
    ).drop(columns=["index"])
    return (simd_spatial_OHdf,)


@app.cell
def _(fit_regularized_ols, simd_spatial_OHdf):
    formula_lag_simd_ros = (
        f"VW_Median_Price ~ {' + '.join(generate_xvar(simd_spatial_OHdf))}"
    )
    ols_lag_params, ols_lag = fit_regularized_ols(
        formula_lag_simd_ros, simd_spatial_OHdf, "VW_Median_Price"
    )
    ols_lag.summary()
    return ols_lag, ols_lag_params


@app.cell
def _(cx, esda, graph, mpatches, np, pd, plt):
    def plot_ols_simd_ros(
        ols_model: "RegressionResultsWrapper",
        OH_gdf: "gpd.GeoDataFrame",
        params: dict[str, float],
        spatial: bool,
        figure_dir: "Path",
        target: str = "VW_Median_Price",
        name: str = "",
    ) -> None:
        """Map actual vs predicted price, residuals and a residual LISA.

        Predicts onto the one-hot frame, then draws four panels (actual,
        predicted, residuals and the combined-weights Moran cluster of the
        residuals) and saves the figure.

        Parameters
        ----------
        ols_model : RegressionResultsWrapper
            Fitted statsmodels OLS results to predict and read residuals from.
        OH_gdf : gpd.GeoDataFrame
            One-hot encoded frame the model was fit on (carries geometry).
        params : dict[str, float]
            Hyperparameters used to fit the model.
        spatial : bool
            Whether this is the spatial-lag model, used for titles/file names.
        figure_dir : Path
            Directory the residual figure is saved into.
        target : str, optional
            Response column being mapped, by default ``"VW_Median_Price"``.
        name : str, optional
            Optional label (e.g. ``"diff"``) for titles and file names.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """
        OH_gdf["Predicted"] = ols_model.predict(OH_gdf)
        OH_gdf["Residuals"] = ols_model.resid

        # Adding wkt for later join because I one-hot encoded the Council_Area
        OH_gdf["geom_wkt"] = OH_gdf.geometry.to_wkt()

        fig, axes = plt.subplots(1, 4, figsize=(25, 7))
        ax = axes.flatten()

        color_map = {
            "High-High": "#2c7bb6",
            "High-Low": "#abd9e9",
            "Low-High": "#fdae61",
            "Low-Low": "#d7191c",
            "Insignificant": "lightgrey",
        }

        # Since provided gdf has multiple duplicated geometries
        # I first need to groupby aggregate the target, Predicted, Residuals and geometry
        # Using volume weighted mean for the Target, Predicted, and Residuals
        # The absolute frame carries Total_Volume, so the duplicate rows (one per
        # funding status) can be volume-weighted. calculate_ros_diff drops the
        # volume columns, so the diff frame has no Total_Volume -- fall back to an
        # unweighted mean there instead of raising KeyError.
        vw_mean = lambda x: np.average(x, weights=OH_gdf.loc[x.index, "Total_Volume"])
        unique_geoms = (
            OH_gdf.groupby("geom_wkt")
            .agg(
                {
                    target: vw_mean,
                    "Predicted": vw_mean,
                    "Residuals": vw_mean,
                    "geometry": "first",
                }
            )
            .set_geometry("geometry")
        )

        contiguity_graph = graph.Graph.build_contiguity(unique_geoms, rook=False)
        knn3_graph = graph.Graph.build_knn(unique_geoms.centroid, k=3)
        combi_graph = graph.Graph.union(contiguity_graph, knn3_graph)
        combi_w = combi_graph.transform("r")

        resid_lisa = esda.Moran_Local(unique_geoms["Residuals"], combi_w, permutations=999)
        unique_geoms["combine_cluster_resid"] = resid_lisa.get_cluster_labels(
            crit_value=0.1
        )

        # Now that I have unique geometries with cluster labels, left join back to original gdf
        OH_gdf = pd.merge(
            OH_gdf, unique_geoms[["combine_cluster_resid"]], on="geom_wkt", how="left"
        )

        price_vmin = min(unique_geoms[target].min(), unique_geoms["Predicted"].min())
        price_vmax = max(unique_geoms[target].max(), unique_geoms["Predicted"].max())

        unique_geoms.plot(
            target,
            legend=True,
            cmap="RdYlGn_r",
            ax=ax[0],
            alpha=0.7,
            vmin=price_vmin,
            vmax=price_vmax,
        )
        ax[0].set_title(f"Actual {target}")

        unique_geoms.plot(
            "Predicted",
            legend=True,
            cmap="RdYlGn_r",
            ax=ax[1],
            alpha=0.7,
            vmin=price_vmin,
            vmax=price_vmax,
        )
        ax[1].set_title(f"Predicted {target}")

        unique_geoms.plot("Residuals", legend=True, cmap="RdBu", ax=ax[2], alpha=0.7)
        ax[2].set_title("Residuals")

        # Colour from unique_geoms (one row per geometry) so the colours line
        # up with the volume weighted mean of the Mortgage and Cash Sales
        # for each Council_Area, instead of falling back to the last row.
        colors = unique_geoms["combine_cluster_resid"].map(color_map)
        unique_geoms.plot(ax=ax[3], color=colors, legend=True, alpha=0.7)
        ax[3].set_title("Residuals Moran Cluster")

        legend_patches = [
            mpatches.Patch(color=hex_val, label=label)
            for label, hex_val in color_map.items()
        ]
        ax[3].legend(
            handles=legend_patches,
            loc="upper left",
            title="Cluster Type",
            frameon=True,
            framealpha=0.9,
        )
        # --- Formatting & Basemaps ---
        for i in range(4):
            cx.add_basemap(ax[i], source="CartoDB DarkMatter", crs=OH_gdf.crs)

        model_kind = "Regularised Spatial Regression" if spatial else "Regularised OLS"
        if params["l1_wt"] == 0:
            regularisation = "Ridge"
        elif params["l1_wt"] == 1:
            regularisation = "Lasso"
        elif 0 < params["l1_wt"] < 0.5:
            regularisation = "Elastic-Ridge"
        elif 0.5 < params["l1_wt"] < 1:
            regularisation = "Elastic-Lasso"
        else:
            regularisation = "Elastic-Net"

        scope_label = f"{name.capitalize()} " if name else ""
        fig.suptitle(
            f"{scope_label}{model_kind} - SIMD on {target} (Regularisation: {regularisation} | alpha={params['alpha']:.3f}, L1={params['l1_wt']:.3f})"
        )

        plt.tight_layout()
        stem = (
            f"ols_residuals_{name + '_' if name else ''}"
            f"{'spatial' if spatial else 'standard'}.png"
        )
        fig.savefig(figure_dir / stem, dpi=150, bbox_inches="tight")
        plt.show()

    return (plot_ols_simd_ros,)


@app.cell
def _(FIGURE, ols, ols_params, plot_ols_simd_ros, simd_concat_OHdf):
    plot_ols_simd_ros(
        ols, simd_concat_OHdf, params=ols_params, spatial=False, figure_dir=FIGURE
    )
    return


@app.cell
def _(FIGURE, ols_lag, ols_lag_params, plot_ols_simd_ros, simd_spatial_OHdf):
    plot_ols_simd_ros(
        ols_lag, simd_spatial_OHdf, params=ols_lag_params, spatial=True, figure_dir=FIGURE
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial Fixed Effects
    """)
    return


@app.cell
def _(DATA_FEATURE, Path, pd, plt, sns):
    def plot_sme_boxplots(
        OH_gdf: "gpd.GeoDataFrame",
        OH_spatial_gdf: "gpd.GeoDataFrame",
        figure_dir: Path,
        name: str = "",
    ) -> None:
        """Compare per-council residual spread for the OLS and spatial models.

        De-dummifies the council columns of both frames, writes each residual
        frame to disk and draws a council-ordered boxplot of residuals split by
        model type.

        Parameters
        ----------
        OH_gdf : gpd.GeoDataFrame
            One-hot frame with residuals from the standard OLS model.
        OH_spatial_gdf : gpd.GeoDataFrame
            One-hot frame with residuals from the spatial regression model.
        figure_dir : Path
            Directory the boxplot figure is saved into.
        name : str, optional
            Optional label (e.g. ``"diff"``) for the title and file name.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """
        dfs_to_concat = []

        # 1. Dictionary to easily label and loop through the dataframes
        datasets = {"Standard OLS": OH_gdf, "Spatial Regression": OH_spatial_gdf}

        prefix = "Council_Area_"

        # 2. Process both datasets into a standard format
        for model_name, gdf in datasets.items():
            # Make a copy so we don't accidentally alter the original geodataframes
            temp_df = gdf.copy()

            dummy_cols = [col for col in temp_df.columns if col.startswith(prefix)]

            temp_df["Council_Area"] = pd.from_dummies(
                temp_df[dummy_cols],
            )
            temp_df["Council_Area"] = temp_df["Council_Area"].str.replace(prefix, "")

            filename = (
                f"Residuals_{name + '_' if name else ''}{model_name}.geojson"
            ).replace(" ", "_")
            if not Path(f"{DATA_FEATURE}/{filename}").exists():
                temp_df.to_file(f"{DATA_FEATURE}/{filename}", driver="GeoJSON")

            temp_df["Model_Type"] = model_name

            dfs_to_concat.append(temp_df[["Council_Area", "Residuals", "Model_Type"]])

        combined_df = pd.concat(dfs_to_concat, ignore_index=True)

        baseline_medians = (
            combined_df[combined_df["Model_Type"] == "Spatial Regression"]
            .groupby("Council_Area")["Residuals"]
            .median()
            .sort_values()
        )
        sorted_councils = baseline_medians.index

        fig, ax = plt.subplots(figsize=(18, 8))

        sns.boxplot(
            data=combined_df,
            x="Council_Area",
            y="Residuals",
            hue="Model_Type",
            order=sorted_councils,
            palette="Set2",
            ax=ax,
            fliersize=2,
        )

        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        title_label = f" ({name})" if name else ""
        ax.set_title(
            f"Comparison of Residuals: Standard OLS vs Spatial Regression{title_label}"
        )
        ax.set_ylabel("Model Residuals")
        ax.set_xlabel("Council Area")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        stem = f"residuals_boxplot{'_' + name if name else ''}.png"
        fig.savefig(figure_dir / stem, dpi=150, bbox_inches="tight")
        plt.show()

    return (plot_sme_boxplots,)


@app.cell
def _(FIGURE, plot_sme_boxplots, simd_concat_OHdf, simd_spatial_OHdf):
    plot_sme_boxplots(simd_concat_OHdf, simd_spatial_OHdf, figure_dir=FIGURE)
    return


@app.cell
def _(simd_concat_OHdf, smf):
    # Council fixed effects: the one-hot Council_Area_* columns are already in
    # generate_xvar, so the "- 1" (no intercept) is what turns them into the full
    # fixed-effect set. The earlier "+ Council_Area" referenced a column that
    # pd.get_dummies had already consumed, which raised a PatsyError.
    fe_ols_xvar = generate_xvar(simd_concat_OHdf)
    fe_ols_xvar = [
        x
        for x in fe_ols_xvar
        if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]
    ]
    formula_fe_ols = f"VW_Median_Price ~ {' + '.join(fe_ols_xvar)} - 1"
    fe_ols = smf.ols(formula_fe_ols, data=simd_concat_OHdf).fit()
    fe_ols.summary()
    return (fe_ols,)


@app.cell
def _(simd_spatial_OHdf, smf):
    fe_spareg_xvar = generate_xvar(simd_spatial_OHdf)
    fe_spareg_xvar = [
        x
        for x in fe_spareg_xvar
        if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]
    ]
    formula_fe_spareg = f"VW_Median_Price ~ {' + '.join(fe_spareg_xvar)} - 1"
    fe_spareg = smf.ols(formula_fe_spareg, data=simd_spatial_OHdf).fit()
    fe_spareg.summary()
    return (fe_spareg,)


@app.cell
def _(FIGURE, cx, pd, plt):
    def plot_fixed_effects(
        fe_ols: "RegressionResultsWrapper",
        fe_spareg: "RegressionResultsWrapper",
        ols_gdf: "gpd.GeoDataFrame",
        spareg_gdf: "gpd.GeoDataFrame",
        name: str = "",
    ) -> None:
        """Map the council fixed effects of the OLS and spatial models.

        Extracts the ``Council_Area`` fixed-effect coefficients from each fitted
        model, joins them onto the de-dummified council geometries and maps the
        two side by side on a shared diverging scale.

        Parameters
        ----------
        fe_ols : RegressionResultsWrapper
            Fitted standard-OLS fixed-effects model.
        fe_spareg : RegressionResultsWrapper
            Fitted spatial-regression fixed-effects model.
        ols_gdf : gpd.GeoDataFrame
            One-hot frame providing geometry for the OLS model.
        spareg_gdf : gpd.GeoDataFrame
            One-hot frame providing geometry for the spatial model.
        name : str, optional
            Optional label (e.g. ``"diff"``) for titles and the file name.

        Returns
        -------
        None
            The figure is saved to disk and shown.
        """
        # 1. Structure data with explicit string names for titles and column headers
        models_data = [
            ("Standard OLS", fe_ols, ols_gdf),
            ("Spatial Regression", fe_spareg, spareg_gdf),
        ]

        # 2. Figure generation OUTSIDE the loop
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        ax = axes.flatten()

        prefix = "Council_Area_"

        for idx, (model_name, model, gdf) in enumerate(models_data):
            # Extract and clean index

            fixed_effects = model.params.filter(like="Council_Area")
            fixed_effects.index = fixed_effects.index.str.replace(
                r"^Q\('(.*)'\)\[T\.True\]$", r"\1", regex=True
            ).str.replace("Council_Area_", "", regex=False)

            max_effect = fixed_effects.abs().max()

            col_name = f"FE_{model_name.replace(' ', '_')}"

            # Forgot to de dummify
            temp_df = gdf.copy()
            dummy_cols = [col for col in temp_df.columns if col.startswith(prefix)]
            temp_df["Council_Area"] = pd.from_dummies(temp_df[dummy_cols])
            temp_df["Council_Area"] = temp_df["Council_Area"].str.replace(prefix, "")

            merged_gdf = temp_df.merge(
                fixed_effects.to_frame(col_name),
                left_on="Council_Area",
                right_index=True,
                how="left",
            )

            merged_gdf.plot(
                col_name,
                legend=True,
                vmin=-max_effect,
                vmax=max_effect,
                cmap="PRGn",
                ax=ax[idx],
                alpha=0.7,
            )

            cx.add_basemap(ax[idx], source="CartoDB DarkMatter", crs=merged_gdf.crs)

            title_label = f" ({name})" if name else ""
            ax[idx].set_title(f"{model_name} Fixed Effects{title_label}")

        plt.tight_layout()
        stem = f"fixed_effects{'_' + name if name else ''}.png"
        fig.savefig(FIGURE / stem, dpi=150, bbox_inches="tight")
        plt.show()

    return (plot_fixed_effects,)


@app.cell
def _(
    fe_ols,
    fe_spareg,
    plot_fixed_effects,
    simd_concat_OHdf,
    simd_spatial_OHdf,
):
    plot_fixed_effects(fe_ols, fe_spareg, simd_concat_OHdf, simd_spatial_OHdf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### SIMD House Price diff OLS
    """)
    return


@app.cell
def _(DIFF_PERIOD, pd):
    def calculate_ros_diff(df: "gpd.GeoDataFrame", diff: DIFF_PERIOD) -> "gpd.GeoDataFrame":
        """Compute the future-minus-past change in SIMD and ROS metrics.

        Selects the past and future year ranges for the requested period and
        differences every weighted-SIMD and volume-weighted price column,
        following the "higher = improved" SIMD convention.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            Joined SIMD/ROS frame spanning all year ranges.
        diff : DIFF_PERIOD
            Period to difference, ``"12-16"`` or ``"16-20"``.

        Returns
        -------
        gpd.GeoDataFrame
            One row per council/funding status with ``diff_*`` change columns.
        """
        # Compare the argument `diff`, not the type alias `DIFF_PERIOD`. The old
        # `DIFF_PERIOD == "12-16"` was always False, so both periods were
        # computed as 2012-2015 -> 2016-2019 and came out identical.
        if diff == "12-16":
            df_past = df[df["Year_Range"] == "2008-2011"]
            df_future = df[df["Year_Range"] == "2012-2015"]
        else:
            df_past = df[df["Year_Range"] == "2012-2015"]
            df_future = df[df["Year_Range"] == "2016-2019"]

        target_cols = [col for col in df_past.columns if col.startswith("weighted")] + [
            "VW_Mean_Price",
            "VW_Median_Price",
        ]

        diff_df = pd.merge(
            df_past,
            df_future,
            on=[
                "Council_Area",
                "Council_Code",
                "geometry",
                "Funding_Status",
            ],
            suffixes=("_past", "_future"),
            how="inner",
        )

        # future - past, matching the SIMD diff convention. SIMD ranks are
        # "higher = better" (less deprived), so a negative value means the
        # metric improved / increased over the period.
        for col in target_cols:
            diff_df["diff_" + col] = diff_df[col + "_future"] - diff_df[col + "_past"]

        diff_df["diff_range"] = diff
        diff_df["Total_Volume"] = diff_df["Total_Volume_future"]

        cols_to_drop = diff_df.filter(regex=r"_past$|_future$").columns
        diff_df = diff_df.drop(columns=cols_to_drop)

        return diff_df

    return (calculate_ros_diff,)


@app.cell
def _(DATA_FEATURE, Path, calculate_ros_diff, gpd, pd, simd_concat_df):
    if Path(f"{DATA_FEATURE}/diff_simd_ros.geojson").exists():
        print("diff_simd_ros.geojson already exists")
        DiffROS_concat_df = gpd.read_file(f"{DATA_FEATURE}/diff_simd_ros.geojson")
    else:
        print("diff_simd_ros.geojson does not exist - creating")
        DiffROS_1216 = calculate_ros_diff(simd_concat_df, diff="12-16")
        DiffROS_1620 = calculate_ros_diff(simd_concat_df, diff="16-20")

        DiffROS_concat_df = pd.concat(
            [DiffROS_1216, DiffROS_1620], axis=0, ignore_index=True
        )
        DiffROS_concat_df.to_file(f"{DATA_FEATURE}/diff_simd_ros.geojson")
    return (DiffROS_concat_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diff OLS

    Same modelling pipeline as the absolute house price, but the response is now
    the per-period change in volume-weighted median price (`diff_VW_Median_Price`)
    modelled on the per-period change.
    """)
    return


@app.cell
def _(DiffROS_concat_df, pd):
    # One hot encoding
    DiffROS_OHdf = pd.get_dummies(
        DiffROS_concat_df,
        columns=["Funding_Status", "diff_range", "Council_Area"],
    )
    return (DiffROS_OHdf,)


@app.cell
def _(DiffROS_OHdf, fit_regularized_ols):
    formula_diff_simd_ros = (
        f"diff_VW_Median_Price ~ {' + '.join(generate_xvar(DiffROS_OHdf))}"
    )
    ols_diff_params, ols_diff = fit_regularized_ols(
        formula_diff_simd_ros, DiffROS_OHdf, "diff_VW_Median_Price"
    )
    ols_diff.summary()
    return ols_diff, ols_diff_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diff Spatial Regression
    """)
    return


@app.cell
def _(DiffROS_concat_df, calculate_combine_lag, pd):
    # Spatial lag of the SIMD-domain diffs, computed per diff period so both
    # 12-16 and 16-20 keep their own neighbour-averaged change values.
    diff_spatial_df = calculate_combine_lag(
        DiffROS_concat_df, period_col="diff_range", xvar_prefix="diff_weighted_"
    )
    diff_spatial_OHdf = pd.get_dummies(
        diff_spatial_df,
        columns=["Funding_Status", "diff_range", "Council_Area"],
    )
    return (diff_spatial_OHdf,)


@app.cell
def _(diff_spatial_OHdf, fit_regularized_ols):
    formula_diff_lag_simd_ros = (
        f"diff_VW_Median_Price ~ {' + '.join(generate_xvar(diff_spatial_OHdf))}"
    )
    ols_diff_lag_params, ols_diff_lag = fit_regularized_ols(
        formula_diff_lag_simd_ros, diff_spatial_OHdf, "diff_VW_Median_Price"
    )
    ols_diff_lag.summary()
    return ols_diff_lag, ols_diff_lag_params


@app.cell
def _(DiffROS_OHdf, FIGURE, ols_diff, ols_diff_params, plot_ols_simd_ros):
    plot_ols_simd_ros(
        ols_diff,
        DiffROS_OHdf,
        params=ols_diff_params,
        spatial=False,
        figure_dir=FIGURE,
        target="diff_VW_Median_Price",
        name="diff",
    )
    return


@app.cell
def _(
    FIGURE,
    diff_spatial_OHdf,
    ols_diff_lag,
    ols_diff_lag_params,
    plot_ols_simd_ros,
):
    plot_ols_simd_ros(
        ols_diff_lag,
        diff_spatial_OHdf,
        params=ols_diff_lag_params,
        spatial=True,
        figure_dir=FIGURE,
        target="diff_VW_Median_Price",
        name="diff",
    )
    return


@app.cell
def _(DiffROS_OHdf, FIGURE, diff_spatial_OHdf, plot_sme_boxplots):
    plot_sme_boxplots(DiffROS_OHdf, diff_spatial_OHdf, figure_dir=FIGURE, name="diff")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial Fixed Effects
    """)
    return


@app.cell
def _(DiffROS_OHdf, smf):
    # Council fixed effects for the diff model: the "- 1" turns the one-hot
    # Council_Area_* columns into the full fixed-effect set. Predicted/Residuals/
    # geom_wkt are dropped because plot_ols_simd_ros adds them to DiffROS_OHdf.
    fe_diff_ols_xvar = generate_xvar(DiffROS_OHdf)
    fe_diff_ols_xvar = [
        x
        for x in fe_diff_ols_xvar
        if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]
    ]
    formula_fe_diff_ols = f"diff_VW_Median_Price ~ {' + '.join(fe_diff_ols_xvar)} - 1"
    fe_diff_ols = smf.ols(formula_fe_diff_ols, data=DiffROS_OHdf).fit()
    fe_diff_ols.summary()
    return (fe_diff_ols,)


@app.cell
def _(diff_spatial_OHdf, smf):
    fe_diff_spareg_xvar = generate_xvar(diff_spatial_OHdf)
    fe_diff_spareg_xvar = [
        x
        for x in fe_diff_spareg_xvar
        if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]
    ]
    formula_fe_diff_spareg = f"diff_VW_Median_Price ~ {' + '.join(fe_diff_spareg_xvar)} - 1"
    fe_diff_spareg = smf.ols(formula_fe_diff_spareg, data=diff_spatial_OHdf).fit()
    fe_diff_spareg.summary()
    return (fe_diff_spareg,)


@app.cell
def _(
    DiffROS_OHdf,
    diff_spatial_OHdf,
    fe_diff_ols,
    fe_diff_spareg,
    plot_fixed_effects,
):
    plot_fixed_effects(
        fe_diff_ols,
        fe_diff_spareg,
        DiffROS_OHdf,
        diff_spatial_OHdf,
        name="diff",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### OLS and spatial regression results

    > Outcomes and statistics might vary slightly when the cells are re-run and models are re-fitted.

    In total, 4 regularised models were fitted against Volume Weighted Median Price and diff Volume Weighted Median Price:
    1. OLS with absolute SIMD and ROS Funding Status
    2. Spatial Regression with absolute SIMD and ROS Funding Status
    3. OLS with relative SIMD and ROS Funding Status
    4. Spatial Regression with relative SIMD and ROS Funding Status

    > Residuals are defined as Actual - Predicted.
    > Overprediction is Actual < Predicted (negative residual); underprediction is Actual > Predicted (positive residual).
    > Critical value is evaluated at 0.1 significance level, more relaxed because of the heavily aggregated nature of the underlying data

    > The mapped residual was calculated using a volume weighted mean across Actual, Predicted, and Residual. This volume weighted mean should be a more accurate reflection of median price because prices because there are significant difference in Mortgage and Cash Sales number in aggregated year_range. This will differ from the median residual seen on the boxplot.

    ##### Absolute

    1. **OLS with absolute SIMD and ROS Funding Status**
       - R2 = 0.924
       - Adjusted R2 = 0.904
       - Regularisation: Ridge (alpha = 0.002, L1 weight = 0)
       - Top 3 SIMD variables, None are statistically significant: Geographic Access (1.303), Health (0.894), Employment (0.683)
       - ROS Funding Status, None are statistically significant: Mortgage Sales (0.186) > Cash Sales (-0.77)
       - Moran cluster of the residuals suggests clustering of Median Price underprediction (High-High) in Inverclyde, East Renfrewshire, Renfrewshire, and Glasgow and overprediction (Low-Low) in Fife, Midlothian, and Borders.

    2. **Spatial Regression with absolute SIMD and ROS Funding Status**
       - R2 = 0.927
       - Adjusted R2 = 0.904
       - Regularisation: Ridge (alpha = 0.002, L1 weight = 0)
       - Top 3 SIMD variables, None are statistically significant: Employment (1.607), Geographic Access (1.424), Health (0.724)
       - Top 3 SIMD lag variables, Health_lag is statistically significant: Health_lag (1.992) (p=0.048), Education_lag (0.575), Geographic Access (0.181)
       - ROS Funding Status, None are statistically significant: Mortgage Sales (0.081) > Cash Sales (-0.128)
       - Moran cluster of the residuals have similar results to OLS model. Underpredicttion (High-High) in Inverclyde, East Renfrewshire, Renfrewshire, Glasgow, with an addition of South Lanarkshire and overprediction (Low-Low) in Fife, Midlothian, and Borders with an addition of East Lothian.

    Reviewing the residual boxplots. There are no obvious observation where the addition of spatial features could reduce the residual variance.

    ##### Relative

    3. **OLS with relative SIMD and ROS Funding Status**
       - R2 = 0.605
       - Adjusted R2 = 0.424
       - Regularisation: Elastic-Lasso (alpha = 0.483, L1 weight = 0.9)
       - Top 3 SIMD variables, None are statistically significant: Employment (1.507), Health (0.78), Housing (-0.250)
       - ROS Funding Status, Cash Sales is statistically significant: Mortgage Sales (0.937) > Cash Sales (-2.173) (p=0.032)
       - Moran cluster of the residuals for the relative OLS model suggest underprediction (High-High) in Glasgow and Orkney and overprediction (Low-Low) for Argyll and Bute.

    4. **Spatial Regression with relative SIMD and ROS Funding Status**
       - R2 = 0.727
       - Adjusted R2 = 0.572
       - Regularisation: Elastic (alpha = 0.08, L1 weight = 0.6)
       - Top 3 SIMD variables, Employment is statistically significant: Employment (1.796) (p=0.076), Health (0.576), Education (0.047)
       - Top 3 SIMD lag variables, None are statistically significant: Health_lag (1.177), Employment_lag (0.631), Education_lag (0.197)
       - ROS Funding Status, None are statistically significant: Mortgage Sales (1.269) > Cash Sales (-0.559)
       - Moran cluster of the residuals suggests similar results for underprediction of Orkney and Glasgow, but overprediction of Argyll and Bute was removed and addition of Edinburgh and Borders.

    Reviewing the resdiual boxplots. The spatial regression for relative models seems to be able to slightly reduce variance for the most overpredicted councils (Borders, Edinburgh, and East Lothian), but increases the variance for those with the smallest median residuals.

    ##### Overall interpretation

    Overall, spatial models do have a slight improvements when compared to non spatial OLS. The first 2 models that fitted the absolute SIMD and ROS values to median price showed high positive correlation when compared to the latter 2 models which fitted the relative diff in SIMD and ROS values to diff in median price.

    Interestingly, in both relative and absolute models. Introduction of spatial lags for SIMD variables seems to change the ordering of the highest t-score, but statistical significance are scarce (only Health_lag for absolute spatial regression and Employment for relative spatial regression). It is inconclusive whether spatial models have any better predictive power when focusing on specific SIMD variables. None of the ROS Funding Status variables are statistically significant either.

    Spatial similarity in significance of residual clusttering between the 2 absolute and spatial relative model suggests that only when spatial features were included in the diff model does it agree with the overprediction. All 4 models have classed absolute and relative underprediction in Glasgow.

    Absolute model underprediction is concetrated in historically deprived councils in West of Scotland, Glasgow, Inverclyde, Renfrewshire, and East Renfrewshire. Aberdeenshire (although not spatially significant) is an anomaly but we saw a sharp anticyclical pattern in drop of SIMD ranks and house prices. While absolute models overprediction are concentrated in East of Scotand, Fife, Edinburgh, Midlothian, and Borders. which are historically more affluent with the exception of Fife.

    While Volume Weighted Median Price have been decreasing for most councils, with the exception of the Islands, and parts of the central belt. The residual and the clustering of the residuals suggests that our model largely overpredicts the reduction in median price. The clustering of the residual suggests underprediction of  Glasgow and Orkney which had a increase in median price. When spatial features were added saw a underpredicted East Lothian which expienced increase in price and underprediction of Highland which saw a decrease in price. Overprediction of the spatial relative models agrees with the absolute models.

    While the models struggle to explain the SIMD variables and ROS variables to be predictive, as there were not a lot of clear agreement between the absolute and relatives, we could agree that spatial features are important when fitting the relatives for better model performance and agreement with the absolute models.
    """)
    return


if __name__ == "__main__":
    app.run()
