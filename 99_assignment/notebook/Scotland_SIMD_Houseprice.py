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
    # Version: 0.0.4              #
    # Date: 2026-06-06            #
    ###############################

    import re
    import sys
    import esda
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as cx
    import seaborn as sns

    from pathlib import Path
    from libpysal import graph
    from typing import Literal, Optional

    sys.path.append("..")
    from src.dictionary import (
        COUNCIL_ALIGNMENT,
        SIMD_DOMAIN_2012,
        SIMD_DOMAIN_2016,
        SIMD_DOMAIN_2020,
    )

    DATA_RAW = Path("../data/01_raw")
    Path("../data/02_intermediate").mkdir(parents=True, exist_ok=True)
    Path("../data/03_feature").mkdir(parents=True, exist_ok=True)
    DATA_INTERMEDIATE = Path("../data/02_intermediate")
    DATA_FEATURE = Path("../data/03_feature")
    return (
        COUNCIL_ALIGNMENT,
        DATA_INTERMEDIATE,
        DATA_RAW,
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
        np,
        pd,
        plt,
        re,
        sns,
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
            simd_pastDF["Council_Area"] = simd_pastDF["Council_Area"].map(
                COUNCIL_ALIGNMENT
            )
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
            simd_pastDF["Council_Area"] = simd_pastDF["Council_Area"].map(
                COUNCIL_ALIGNMENT
            )
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

    return (simd_preprocessing,)


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
                simd_data_past=Path(
                    f"{DATA_RAW}/simd2016_withgeog/simd2016_withinds.csv"
                ),
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
        print(
            "SIMD data already exists - skip preprocessing and read from intermediate"
        )
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

        cx.add_basemap(ax[idx], crs=df.crs, source="CartoDB DarkMatter")

    plt.tight_layout()
    plt.show()
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
    def plot_k_elbow(year: int, dissolve: bool) -> plt.Figure:
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
        plt.show()

        return fig

    return (plot_k_elbow,)


@app.cell
def _(plot_k_elbow):
    k_2012 = plot_k_elbow(2012, dissolve=True)
    k_2016 = plot_k_elbow(2016, dissolve=True)
    k_2020 = plot_k_elbow(2020, dissolve=True)

    k_2012
    return k_2016, k_2020


@app.cell
def _(k_2016):
    k_2016
    return


@app.cell
def _(k_2020):
    k_2020
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
    ):
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
                if col.startswith("diff")
                and col.endswith("_Rank")
                and col not in meta_cols
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
            lisa_queen = esda.Moran_Local(domain_df[col], contiguity_r, permutations=99)
            lisa_knn3 = esda.Moran_Local(domain_df[col], knn3_r, permutations=99)
            lisa_combine = esda.Moran_Local(domain_df[col], combi_w, permutations=99)
            domain_df[f"combine_moran'sI_{col}"] = lisa_combine.Is
            domain_df[f"combine_pvalue_{col}"] = lisa_combine.p_sim
            # Due to spatial aggregation, replaxing the p-value
            domain_df[f"queen_cluster_{col}"] = lisa_queen.get_cluster_labels(
                crit_value=0.1
            )
            domain_df[f"knn3_cluster_{col}"] = lisa_knn3.get_cluster_labels(
                crit_value=0.1
            )
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
    cluster_Dissolve2012
    return cluster_Dissolve2012, cluster_Dissolve2016, cluster_Dissolve2020


@app.cell
def _(Literal, cx, plt, sns):
    GRAPH_TYPE = Literal["knn3", "queen", "combine"]

    def plot_lisa(graph_df, method):
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
        return fig

    return (plot_lisa,)


@app.cell
def _(
    cluster_Dissolve2012,
    cluster_Dissolve2016,
    cluster_Dissolve2020,
    plot_lisa,
):
    combine2012_moran = plot_lisa(cluster_Dissolve2012, method="combine")
    combine2016_moran = plot_lisa(cluster_Dissolve2016, method="combine")
    combine2020_moran = plot_lisa(cluster_Dissolve2020, method="combine")
    combine2012_moran
    return combine2016_moran, combine2020_moran


@app.cell
def _(combine2016_moran):
    combine2016_moran
    return


@app.cell
def _(combine2020_moran):
    combine2020_moran
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

    def plot_abs_diff(simd_mergeDFlong, simd_diffDFlong):

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
                legend=(idx==0),
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
                fontsize='small'
                )

        plt.tight_layout()
        plt.show()

    plot_abs_diff(simd_mergeDFlong, simd_diffDFlong)
    return


@app.cell
def _(SIMDdiff_1216, SIMDdiff_1620, moran_local, plot_lisa):
    diffcluster_SIMD1216 = moran_local(
        year=None, dissolve=True, diff=True, diff_df=SIMDdiff_1216
    )
    diffcluster_SIMD1620 = moran_local(
        year=None, dissolve=True, diff=True, diff_df=SIMDdiff_1620
    )
    combinediff_1216moran = plot_lisa(diffcluster_SIMD1216, method="combine")
    combinediff_1620moran = plot_lisa(diffcluster_SIMD1620, method="combine")
    combinediff_1216moran
    return combinediff_1620moran, diffcluster_SIMD1216


@app.cell
def _(diffcluster_SIMD1216):
    diffcluster_SIMD1216
    return


@app.cell
def _(combinediff_1620moran):
    combinediff_1620moran
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 1: Scotland SIMD preprocessing summary

    I have rolled-up the SIMD data into Council Areas, using a volume weighted approach for each year. I calculated the Moran's I foir each domain using a lower critical level of 0.1

    For absolute numbers:
    - For all periods, the health domain consistently has the highest spatial autocorrelation, meanwhile, Income and Employment have experience some significance in spatial autocorrelation in year 2012 and 2016, but not in 2020.
    - For period 2012 and 2016. There are more counts of High-High clustering in Health, Income and Employment than in other domains, however in 2020, Low-Low clustering has gained grounds, particularly in Geographic Access and Crime.
    - The Councils of Highlands, Shetlands, and Orkney Islands have the lowest spatial autocorrelation in Health, Income and Employment domains, being above average and significance in their neighbouring affects.
    - Looking at the absolute lineplots, its clear that there are no significant differences in the outcomes of rankings between the years.

    For relative differences:
    - This is a significantly more interesting metric. As it shows that the spatio autocorrelation of Health, Income and Employment has less significance compared to the absolute rankings. Suggesting that changes in these domains are less sensitive to spatial effects than the absolute rankings.
    - Meanwhile for Crime 2016-2020, relative differences in Low-Low clustering have gain grounds, suggesting lower than average changes begets lower changes in crime in the surround areas mainly around South-East Scotland. But this was not signficant in absolute ranking.
    - The largest change came from Health, and particularly Income and Employment in period 2016-2020. Where we saw massive gains in relative spatial autocorrelatiojn in these domains, driven predominantly by High-High and Low-Low clustering. Where Central and Southern Scotland saw above average improvements and also in their neighbouring areas.
    - Aberdeenshire is interesting as well, where in employment domain, it was higher than average surrounding by higher areas in Health, Income, and Employment. However, relative differences suggest that the improvement in Health and Employment was only for 2012-2016, but decrease to below average change in 2016-2020.
    - While for the Highlands, Shetlands, and Orkney Islands, although aboslute rankings suggests they are consistently above average for Income and Employment, relative differences suggest that they have been experiencing negative changes with signficance negatives changes in the surrounding areas. I.e. quality of life for Income and Employment domain are high but worsening.

    The result might indicate that the more affluent rural areas in the north of Scotland with strong autocorrelation might be experience negative changes recently. While more economically deprived areas of Western Central Belt have not experienced much changes at all. with the rest of the central belt being a mixed bag results. The borders and Dumfries and Galloway are also inconsisntent. Excluding spatial autocorrelation, we generally saw stagnation with 2016-2020 saw lower positive changes in improvement compared to 2012-2016. Growth in quality of life have generally reduced, with Crime seeing mixed result. Most significant was housing where there were little to no changes in ranking for 2016-2020. Employment and Income were both trending into the negative with 2016-2020, suggesting that income and employment metrics have worsen overall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
