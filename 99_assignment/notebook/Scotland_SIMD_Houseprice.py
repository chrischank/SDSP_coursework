import marimo

__generated_with = "0.23.9"
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
    # Notebook for SDSP Assessment#
    # Maintainer: Christopher Chan#
    # Version: 0.0.10             #
    # Date: 2026-06-16            #
    ###############################

    import re
    import sys
    import esda
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import contextily as cx
    import seaborn as sns
    import statsmodels.formula.api as smf

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
        mpatches,
        np,
        pd,
        plt,
        re,
        smf,
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
    FIGURE,
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
    fig.savefig(FIGURE / "simd_income_maps.png", dpi=150, bbox_inches="tight")
    #plt.show()
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
    def plot_k_elbow(year: int, dissolve: bool, figure_dir) -> plt.Figure:
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

    def plot_lisa(graph_df, method: GRAPH_TYPE, name, figure_dir):
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
        fig.savefig(
            figure_dir / f"lisa_{method}_{name}.png", dpi=150, bbox_inches="tight"
        )
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
    combinediff_1216moran
    return (combinediff_1620moran,)


@app.cell
def _(combinediff_1620moran):
    combinediff_1620moran
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 1: SIMD preprocessing summary

    I have rolled-up the SIMD data into Council Areas, using a volume weighted approach for each year. I calculated the Moran's I foir each domain using a lower critical level of 0.1

    For absolute numbers:
    - For all periods, the health domain consistently has the highest statistical significance in spatial autocorrelation, meanwhile, Income and Employment have experience some significance in spatial autocorrelation in year 2012 and 2016, but reduced in 2020.
    - For period 2012 and 2016. There are more counts of High-High clustering in Health, Income and Employment than in other meaning that for Council Areas that are statistically significant, spatial effects are more pronounce when it comes to health, income, and employment when the council is above average and surrounding by areas of above average. However in 2020, Low-Low clustering has gained grounds, particularly in Crime. These councils are concentrated in Central Scotland.
    - The Councils of Highlands, Shetlands, and Orkney Islands have the highest clustering for Health, Income, Crime and Employment domains, being above average and significance in their neighbouring affects.

    For relative differences:
    - This is a significantly more interesting metric. Looking at the relative lineplots, we saw a smaller increase of ranking for Education and Geographic Access for 2016-2020 when compared to 2012-2016, while many other domains experienced a reduction in rank for most councils. Looking at cluster for diff for Highlands, Shetlands, and Orkney Islands, although aboslute rankings suggests they are consistently above average for Income and Employment, relative differences suggest that high absolute spatial relationship do not translate to relative. Particularly for Income and Employment, The Shetland and Orkeny Islands experiencing negative changes with signficance negatives changes in the surrounding areas. I.e. quality of life for Income and Employment domain are high but worsening.
    - Meanwhile for Income, Employment, and Crime 2012-2016, relative differences in Low-Low clustering have gain grounds in the borders of Scotland, suggesting lower than average changes begets lower changes in these domains in the surround areas. But this was not signficant in absolute ranking for 2012 and 2016. This might suggest that the border regions are experiencing increase in crime, and negative income and employment outcome caused by neighbouring affects temporarily.
    - The largest change came from Crime, and particularly Income and Employment in period 2012-2026. With income and employment domain experiencing further signicant spatial clustering in 2016-2020. Both Low-Low clustering and High-High clustering gaining grounds. But the council areas difference could not conclude whether there were spatial-temporal affects carried over from previous periods.
    - Housing in the highland council were consistently high for 2016 and 2020 absolute with signficance in High-High clustering. However, relative differences clustering have worsen from Low-High to Low-Low for 2012-2016, 2016-2020 respectively. This might indicate that although the surrounding regions still have high housing availability, this might suggest that Housing availability for highlands is worsening.
    - Aberdeen, Moray and Aberdeenshire is interesting as well, where in Income, and Employment domain, it was higher than average surrounding by higher areas consistently. However, relative differences in clustering suggest that the improvement in Health and Employment was only for 2012-2016, but decrease to below average change and clustering (Low-Low) in 2016-2020.

    The result might indicate that the more affluent rural areas in the north of Scotland with strong autocorrelation might be experience negative changes recently. While more economically deprived areas of Western Scotland (Ayrshire) have not experienced much changes at all. with the rest of the central belt being a mixed bag results, with central scotland experiencing improvement in Income and Employment. The borders and Dumfries and Galloway are also inconsisntent. Excluding spatial autocorrelation, we generally saw stagnation with 2016-2020 saw lower positive changes in improvement or worsening compared to 2012-2016. Growth in quality of life have generally reduced, with Crime seeing mixed result. Most significant was housing where there were little to no changes in ranking for 2016-2020. Employment and Income were both trending into the negative with 2016-2020, suggesting that income and employment metrics have worsen overall.
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
def _(INFLATION_2011, gpd, pd):
    def ros_volume_weighted(
        df: pd.DataFrame, simd_year: int, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

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

        def _calculate_weighted_metrics(group):
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
def _(ros_concat_df):
    ros_concat_df.sample(8)
    return


@app.cell
def _(cx, np, plt):
    # Lets plot and explore the volume weighted average price for different funding status
    def plot_house_price(vw_gdf, average: str, figure_dir):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        ax = axes.flatten()
        idx = 0

        min_price = np.nanmin(vw_gdf[["VW_Mean_Price", "VW_Median_Price"]].to_numpy())
        max_price = np.nanmax(vw_gdf[["VW_Mean_Price", "VW_Median_Price"]].to_numpy())

        for status in vw_gdf["Funding_Status"].unique():
            for year in vw_gdf["Year_Range"].unique():
                filter_gdf = vw_gdf[
                    (vw_gdf["Year_Range"] == year)
                    & (vw_gdf["Funding_Status"] == status)
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
        fig.savefig(
            figure_dir / f"house_price_{average}.png", dpi=150, bbox_inches="tight"
        )
        plt.show()

    return (plot_house_price,)


@app.cell
def _(FIGURE, plot_house_price, ros_concat_df):
    plot_house_price(ros_concat_df, "VW_Median_Price", figure_dir=FIGURE)
    return


@app.cell
def _(FIGURE, plot_house_price, ros_concat_df):
    plot_house_price(ros_concat_df, "VW_Mean_Price", figure_dir=FIGURE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling
    1. First I will perform an OLS modelling and look at the residual to see the results for absolute and differences
    2. Then I will do the same with spatial lag added to both dataset. The combined contiguity + KNN3 lag is built per period (year for the absolute model, diff-period for the diff model) so every council-period keeps its own neighbour average and the models are fit on the full panel.
    3. Once I have a performant model, I will use the rest of the ros data to predict the next set of simd domain

    #### OLS modelling
    """)
    return


@app.cell
def _(DATA_FEATURE, gpd, pd, re):
    def ols_preprocessing(df_2012, df_2016, df_2020, ros_df):
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

    simd_concat_df
    return (simd_concat_df,)


@app.cell
def _(pd, simd_concat_df):
    # One-hot encoding of cateogorical variables
    simd_concat_OHdf = pd.get_dummies(
        simd_concat_df,
        columns=["Year_Range", "Funding_Status", "Council_Area"],
    ).drop(columns=["index"])
    simd_concat_OHdf.head()
    return (simd_concat_OHdf,)


@app.function
def generate_xvar(df, price_prefix="VW"):
    # Identifiers / geometry are never predictors. Total_Volume only exists in
    # the absolute frame, so drop only what is present (the diff frame omits it).
    drop_cols = [
        col
        for col in ["Council_Code", "Total_Volume", "geometry"]
        if col in df.columns
    ]

    # Drop the price response columns - the raw "VW_*" set for the absolute
    # model and the "diff_VW_*" set for the diff model (both the modelled median
    # and the mean we are not modelling) so they never leak in as predictors.
    price_cols = [
        col
        for col in df.columns
        if col.startswith(price_prefix) or col.startswith(f"diff_{price_prefix}")
    ]

    ros_X_var = df.columns.drop(drop_cols + price_cols)
    ros_X_var = [f"Q('{col}')" for col in ros_X_var]

    return ros_X_var


@app.cell
def _(simd_concat_OHdf, smf):
    formula_simd_ros = (
        f"VW_Median_Price ~ {' + '.join(generate_xvar(simd_concat_OHdf))}"
    )
    ols = smf.ols(formula_simd_ros, data=simd_concat_OHdf).fit()
    ols.summary()
    return (ols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### OLS modelling (Spatial)
    """)
    return


@app.cell
def _(graph, pd):
    def calculate_combine_lag(
        concat_df, group_col="Council_Code", period_col=None, xvar_prefix="weighted_"
    ):
        # Columns to calculate spatial lag: weighted_* (absolute) or diff_weighted_* (diff),
        # Funding_Status_* (absolute) or diff_Funding_Status_* (diff)
        xvar_cols = [col for col in concat_df.columns if col.startswith(xvar_prefix)]

        base_geoms = (
            concat_df.groupby(group_col)
            .agg({"geometry": "first"})
            .set_geometry("geometry")
        )
        contiguity_graph = graph.Graph.build_contiguity(base_geoms)
        knn3_graph = graph.Graph.build_knn(base_geoms.centroid, k=3)
        combi_w = graph.Graph.union(contiguity_graph, knn3_graph).transform("r")

        lag_cols = [f"{col}_lag" for col in xvar_cols]

        periods = (
            [None] if period_col is None else list(concat_df[period_col].unique())
        )
        merge_keys = [group_col] if period_col is None else [group_col, period_col]

        lag_frames = []
        for period in periods:
            sub = (
                concat_df
                if period is None
                else concat_df[concat_df[period_col] == period]
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

    simd_spatial_OHdf.columns
    return (simd_spatial_OHdf,)


@app.cell
def _(simd_spatial_OHdf, smf):
    formula_lag_simd_ros = (
        f"VW_Median_Price ~ {' + '.join(generate_xvar(simd_spatial_OHdf))}"
    )
    ols_lag = smf.ols(formula_lag_simd_ros, data=simd_spatial_OHdf).fit()
    ols_lag.summary()
    return (ols_lag,)


@app.cell
def _(cx, esda, graph, mpatches, pd, plt):
    def plot_ols_simd_ros(
        ols_model, OH_gdf, spatial: bool, figure_dir, target="VW_Median_Price", name=""
    ):
        OH_gdf["Predicted"] = ols_model.predict(OH_gdf)
        OH_gdf["Residuals"] = ols_model.resid

        # Adding wkt for later join because I one-hot encoded the Council_Area
        OH_gdf["geom_wkt"] = OH_gdf.geometry.to_wkt()

        price_vmin = min(OH_gdf[target].min(), OH_gdf["Predicted"].min())
        price_vmax = max(OH_gdf[target].max(), OH_gdf["Predicted"].max())

        fig, axes = plt.subplots(1, 4, figsize=(25, 7))
        ax = axes.flatten()

        color_map = {
            "High-High": "#2c7bb6",
            "High-Low": "#abd9e9",
            "Low-High": "#fdae61",
            "Low-Low": "#d7191c",
            "Insignificant": "lightgrey",
        }

        OH_gdf.plot(
            target,
            legend=True,
            cmap="RdYlGn_r",
            ax=ax[0],
            alpha=0.7,
            vmin=price_vmin,
            vmax=price_vmax,
        )
        ax[0].set_title(f"Actual {target}")

        OH_gdf.plot(
            "Predicted",
            legend=True,
            cmap="RdYlGn_r",
            ax=ax[1],
            alpha=0.7,
            vmin=price_vmin,
            vmax=price_vmax,
        )
        ax[1].set_title(f"Predicted {target}")

        OH_gdf.plot("Residuals", legend=True, cmap="RdBu", ax=ax[2], alpha=0.7)
        ax[2].set_title("Residuals")

        # Since provided gdf has multiple duplicated geometries
        # I first need to groupby aggregate the residuals and geometry
        unique_geoms = (
            OH_gdf.groupby("geom_wkt")
            .agg(
                {
                    "Residuals": "mean",
                    "geometry": "first",
                }
            )
            .set_geometry("geometry")
        )

        contiguity_graph = graph.Graph.build_contiguity(unique_geoms, rook=False)
        knn3_graph = graph.Graph.build_knn(unique_geoms.centroid, k=3)
        combi_graph = graph.Graph.union(contiguity_graph, knn3_graph)
        combi_w = combi_graph.transform("r")

        resid_lisa = esda.Moran_Local(
            unique_geoms["Residuals"], combi_w, permutations=99
        )
        unique_geoms["combine_cluster_resid"] = resid_lisa.get_cluster_labels(
            crit_value=0.1
        )

        # Now that I have unique geometries with cluster labels, left join back to original gdf
        OH_gdf = pd.merge(
            OH_gdf, unique_geoms[["combine_cluster_resid"]], on="geom_wkt", how="left"
        )

        colors = OH_gdf["combine_cluster_resid"].map(color_map)
        OH_gdf.plot(
            "combine_cluster_resid", ax=ax[3], color=colors, legend=True, alpha=0.7
        )
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

        title_suffix = "(Spatial Regression)" if spatial else "(Standard OLS)"
        model_label = f"{name} " if name else ""
        fig.suptitle(
            f"{model_label}OLS Regression of SIMD on {target} - {title_suffix}"
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
def _(FIGURE, ols, plot_ols_simd_ros, simd_concat_OHdf):
    plot_ols_simd_ros(ols, simd_concat_OHdf, spatial=False, figure_dir=FIGURE)
    return


@app.cell
def _(FIGURE, ols_lag, plot_ols_simd_ros, simd_spatial_OHdf):
    plot_ols_simd_ros(ols_lag, simd_spatial_OHdf, spatial=True, figure_dir=FIGURE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Spatial Fixed Effects
    """)
    return


@app.cell
def _(DATA_FEATURE, Path, pd, plt, sns):
    def plot_sme_boxplots(OH_gdf, OH_spatial_gdf, figure_dir, name=""):
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
    fe_ols_xvar = [x for x in fe_ols_xvar if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]]
    formula_fe_ols = (
        f"VW_Median_Price ~ {' + '.join(fe_ols_xvar)} - 1"
    )
    fe_ols = smf.ols(formula_fe_ols, data=simd_concat_OHdf).fit()
    fe_ols.summary()
    return (fe_ols,)


@app.cell
def _(simd_spatial_OHdf, smf):
    fe_spareg_xvar = generate_xvar(simd_spatial_OHdf)
    fe_spareg_xvar = [x for x in fe_spareg_xvar if x not in ["Q('Predicted')", "Q('Residuals')", "Q('geom_wkt')"]]
    formula_fe_spareg = (
        f"VW_Median_Price ~ {' + '.join(fe_spareg_xvar)} - 1"
    )
    fe_spareg = smf.ols(formula_fe_spareg, data=simd_spatial_OHdf).fit()
    fe_spareg.summary()
    return (fe_spareg,)


@app.cell
def _(FIGURE, cx, pd, plt):
    def plot_fixed_effects(fe_ols, fe_spareg, ols_gdf, spareg_gdf, name=""):
        # 1. Structure data with explicit string names for titles and column headers
        models_data = [
            ("Standard OLS", fe_ols, ols_gdf),
            ("Spatial Regression", fe_spareg, spareg_gdf)
        ]

        # 2. Figure generation OUTSIDE the loop
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        ax = axes.flatten()

        prefix = "Council_Area_"

        for idx, (model_name, model, gdf) in enumerate(models_data):
            # Extract and clean index

            fixed_effects = model.params.filter(like="Council_Area")
            fixed_effects.index = (
                fixed_effects.index
                .str.replace(r"^Q\('(.*)'\)\[T\.True\]$", r"\1", regex=True)
                .str.replace("Council_Area_", "", regex=False)
            )

            max_effect = fixed_effects.abs().max()

            col_name = f"FE_{model_name.replace(' ', '_')}"

            # Forgot to de dummify
            temp_df = gdf.copy() 
            dummy_cols = [col for col in temp_df.columns if col.startswith(prefix)]
            temp_df["Council_Area"] = pd.from_dummies(
                temp_df[dummy_cols]
            )
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
                alpha=0.7
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
    def calculate_ros_diff(df, diff: DIFF_PERIOD):
        # Compare the argument `diff`, not the type alias `DIFF_PERIOD`. The old
        # `DIFF_PERIOD == "12-16"` was always False, so both periods were
        # computed as 2012-2015 -> 2016-2019 and came out identical.
        if diff == "12-16":
            df_past = df[df["Year_Range"] == "2008-2011"]
            df_future = df[df["Year_Range"] == "2012-2015"]
        else:
            df_past = df[df["Year_Range"] == "2012-2015"]
            df_future = df[df["Year_Range"] == "2016-2019"]

        target_cols = [
            col for col in df_past.columns if col.startswith("weighted")
        ] + ["VW_Mean_Price", "VW_Median_Price"]

        diff_df = pd.merge(
            df_past,
            df_future,
            on=["Council_Area", "Council_Code", "geometry", "Funding_Status"],
            suffixes=("_past", "_future"),
        )

        # future - past, matching the SIMD diff convention. SIMD ranks are
        # "higher = better" (less deprived), so a positive value means the
        # metric improved / increased over the period.
        for col in target_cols:
            diff_df["diff_" + col] = diff_df[col + "_future"] - diff_df[col + "_past"]

        diff_df["diff_range"] = diff

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

    DiffROS_concat_df
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
    DiffROS_OHdf.head()
    return (DiffROS_OHdf,)


@app.cell
def _(DiffROS_OHdf, smf):
    formula_diff_simd_ros = (
        f"diff_VW_Median_Price ~ {' + '.join(generate_xvar(DiffROS_OHdf))}"
    )
    ols_diff = smf.ols(formula_diff_simd_ros, data=DiffROS_OHdf).fit()
    ols_diff.summary()
    return (ols_diff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diff OLS (Spatial)
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

    diff_spatial_OHdf.head()
    return (diff_spatial_OHdf,)


@app.cell
def _(diff_spatial_OHdf, smf):
    formula_diff_lag_simd_ros = (
        f"diff_VW_Median_Price ~ {' + '.join(generate_xvar(diff_spatial_OHdf))}"
    )
    ols_diff_lag = smf.ols(formula_diff_lag_simd_ros, data=diff_spatial_OHdf).fit()
    ols_diff_lag.summary()
    return (ols_diff_lag,)


@app.cell
def _(DiffROS_OHdf, FIGURE, ols_diff, plot_ols_simd_ros):
    plot_ols_simd_ros(
        ols_diff,
        DiffROS_OHdf,
        spatial=False,
        figure_dir=FIGURE,
        target="diff_VW_Median_Price",
        name="diff",
    )
    return


@app.cell
def _(FIGURE, diff_spatial_OHdf, ols_diff_lag, plot_ols_simd_ros):
    plot_ols_simd_ros(
        ols_diff_lag,
        diff_spatial_OHdf,
        spatial=True,
        figure_dir=FIGURE,
        target="diff_VW_Median_Price",
        name="diff",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diff Spatial Fixed Effects
    """)
    return


@app.cell
def _(DiffROS_OHdf, FIGURE, diff_spatial_OHdf, plot_sme_boxplots):
    plot_sme_boxplots(DiffROS_OHdf, diff_spatial_OHdf, figure_dir=FIGURE, name="diff")
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
    formula_fe_diff_spareg = (
        f"diff_VW_Median_Price ~ {' + '.join(fe_diff_spareg_xvar)} - 1"
    )
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

    In total, 4 models were fitted against Volume Weighted Median Price:
    1. OLS with absolute SIMD and ROS Funding Status
    2. Spatial Regression with aboslute SIMD and ROS Funding Status
    3. OLS with relative SIMD and ROS Funding Status
    4. Spatial Regression with relative SIMD and ROS Funding Status

    ##### Absolute
    1. OLS withy absolute SIMD and ROS Funding Status: \
    R2 = 0.938 \
    Adjusted R2 = 0.928 \
    Top 3 SIMD variables: Health (0.817), Employment (0.711), Income (0.673) \
    ROS Funding Status: Mortgage Sales (2.015) > All properties (1.455) > Cash Sales (-0.079)

    2. Spatial Regression with absolute SIMD and ROS Funding Status: \
    R2 = 0.944 \
    Adjusted R2 = 0.933 \
    Top 3 SIMD variables: Education (1.578), Income (1.303), Geographic Access (0.963) \
    Top 3 SIMD lag variables: Health_lag (3.406), Education_lag (2.286), Income_lag (1.419) \
    ROS Funding Status: Mortgage Sales (-2.162) > All properties (-2.234) > Cash Sales (-2.433) \

    ##### Relative
    3. OLS with relative SIMD and ROS Funding Status: \
    R2 = 0.719 \
    Adjusted R2 = 0.645 \
    Top 3 SIMD variables: Employment (3.323), Health (1.290), Housing (-0.138) \
    ROS Funding Status: Mortgage Sales (1.757) > All properties (-1.002) > Cash Sales (-3.861) \

    4. Spatial Regression with relative SIMD and ROS Funding Status: \
    R2 = 0.754 \
    Adjusted R2 = 0.676 \
    Top 3 SIMD variables: Employment (3.844), Health (1.508), Education (-0.312) \
    Top SIMD lag variables: Employment_lag (2.406), Education_lag (1.792), Health_lag (1.295) \
    ROS Funding Status: Mortgaage Sales (2.663) > All properties (0.783) > Cash Sales (-1.164) \
    """)
    return


if __name__ == "__main__":
    app.run()
