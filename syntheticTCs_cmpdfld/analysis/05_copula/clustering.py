import numpy as np
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

mpl.use('TkAgg')
plt.ion()

###################################################################################################################
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\01_return_period_tables')

ncep_csvfiles = [f for f in os.listdir() if f.endswith('ncep.csv')]
histdf = pd.concat((pd.read_csv(file, index_col=0) for file in ncep_csvfiles), ignore_index=False)

# Track info at landfall
track_table_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\tracks\ncep_landfall_track_info.csv'
track_df = pd.read_csv(track_table_filepath, index_col=0)
track_df.set_index('tc_id', inplace=True, drop=True)
track_df_hist = track_df[['rmw100','pstore100','speed100','vstore100']]

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\07_correlation_matrix\cluster_PCA')
###################################################################################################################


cluster_by_floodtype = False
if cluster_by_floodtype is True:
    df = histdf[histdf['basin'] == 'Domain']
    df = pd.concat([df, track_df_hist], axis=1)
    cluster_labels_dict = {}  # To store cluster labels per flood type
    flood_types = ['Coastal_Area_sqkm', 'Runoff_Area_sqkm', 'Compound_Area_sqkm']

    for flood_type in flood_types:
        print(f"\n=== Processing {flood_type} ===")

        pct90 = df[flood_type].quantile(0.9)
        subset = df[df[flood_type] >= pct90].copy()

        features = subset[['rmw100', 'speed100', 'vstore100']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        sse = []
        sil_scores = []
        K_range = range(2, 8)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            sse.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(features_scaled, labels))

        # Plot Elbow and Silhouette
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(K_range, sse, 'bo-')
        plt.title(f'Elbow (SSE) - {flood_type}')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')

        plt.subplot(1, 2, 2)
        plt.plot(K_range, sil_scores, 'go-')
        plt.title(f'Silhouette Score - {flood_type}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'SSE_silhouetteScore_{flood_type}.png')
        plt.close()

        k_optimal = K_range[np.argmax(sil_scores)]
        print(f"Optimal clusters for {flood_type}: {k_optimal}")

        kmeans_final = KMeans(n_clusters=k_optimal, random_state=42)
        subset['cluster'] = kmeans_final.fit_predict(features_scaled)

        # Save cluster labels back (use storm IDs or index)
        cluster_labels_dict[flood_type] = subset['cluster']

        # Cluster profiles
        profile = subset.groupby('cluster')[['rmw100', 'speed100', 'vstore100', flood_type]].mean().round(2)
        print("\nCluster profile (means):")
        print(profile)

        # Boxplots
        plt.figure(figsize=(12, 4))
        for i, feature in enumerate(['rmw100', 'speed100', 'vstore100'], 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x='cluster', y=feature, data=subset)
            plt.title(f'{feature} by Cluster ({flood_type})')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'boxplot_clusters_{flood_type}.png')
        plt.close()

        # PCA visualization
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(features_scaled)
        subset.loc[:, 'pca1'] = pca_components[:, 0]
        subset.loc[:, 'pca2'] = pca_components[:, 1]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='pca1', y='pca2',
            hue='cluster',
            palette='tab10',
            size=flood_type,
            sizes=(20, 200),
            alpha=0.7,
            data=subset
        )
        plt.title(f'PCA Clusters for {flood_type} (size by flood extent)')
        plt.show()
        plt.savefig(f'pcs_clusters_{flood_type}.png')
        plt.close()

        # Kruskal-Wallis test
        print("\nKruskal-Wallis tests:")
        for col in ['rmw100', 'speed100', 'vstore100', flood_type]:
            groups = [group[col].values for _, group in subset.groupby('cluster')]
            stat, p = kruskal(*groups)
            print(f"{col}: H={stat:.2f}, p={p:.4g}")

        # Pairwise Mann-Whitney U tests between clusters
        print("\nPairwise Mann-Whitney U tests:")
        clusters = subset['cluster'].unique()
        for col in ['rmw100', 'speed100', 'vstore100', flood_type]:
            print(f"\nFeature: {col}")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    group1 = subset[subset['cluster'] == clusters[i]][col]
                    group2 = subset[subset['cluster'] == clusters[j]][col]
                    U, pval = mannwhitneyu(group1, group2, alternative='two-sided')
                    print(f"Cluster {clusters[i]} vs {clusters[j]}: U={U:.1f}, p={pval:.4g}")


    for flood_type, labels in cluster_labels_dict.items():
        # This aligns by index
        df.loc[labels.index, f'{flood_type}_cluster'] = labels

    # For visualization, select storms that have clusters in all flood types
    common_index = df.dropna(subset=[f'{ft}_cluster' for ft in flood_types]).index
    df_vis = df.loc[common_index]

    # Create a PCA embedding on the standardized storm features
    features = df_vis[['rmw100', 'speed100', 'vstore100']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(features_scaled)

    df_vis['pca1'] = pca_comp[:, 0]
    df_vis['pca2'] = pca_comp[:, 1]

    # Plot PCA colored by clusters from each flood type side-by-side
    plt.figure(figsize=(18,5))
    for i, flood_type in enumerate(flood_types, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(
            data=df_vis,
            x='pca1', y='pca2',
            hue=f'{flood_type}_cluster',
            palette='tab10',
            alpha=0.7,
            legend='full'
        )
        plt.title(f'Clusters by {flood_type} (PCA space)')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'clusters_sidebyside_{flood_type}')
    plt.close()

    print("Adjusted Rand Index between clusterings:")

    for i in range(len(flood_types)):
        for j in range(i+1, len(flood_types)):
            c1 = df[f'{flood_types[i]}_cluster'].dropna()
            c2 = df[f'{flood_types[j]}_cluster'].dropna()
            # Use intersection of indices to compare same storms
            common = c1.index.intersection(c2.index)
            ari = adjusted_rand_score(c1.loc[common], c2.loc[common])
            print(f"{flood_types[i]} vs {flood_types[j]}: ARI = {ari:.3f}")


cluster_by_basin = True
if cluster_by_basin is True:
    watersheds = histdf['basin'].unique()
    results = {}
    for ws in watersheds:
        print(f"\n=== Watershed: {ws} ===")

        df_ws = histdf[histdf['basin'] == ws].copy()
        df_ws = pd.concat([df_ws, track_df_hist], axis=1)

        # Subset storms exceeding 90th percentile Compound_Area in this watershed
        pct90_compound = df_ws['Compound_Area_sqkm'].quantile(0.9)
        df_ws_sub = df_ws[df_ws['Compound_Area_sqkm'] >= pct90_compound].copy()

        features = df_ws_sub[['rmw100', 'speed100', 'vstore100']]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Find optimal k with silhouette (optional elbow)
        K_range = range(2, 8)
        sil_scores = []
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            sil_scores.append(silhouette_score(features_scaled, labels))

        optimal_k = K_range[np.argmax(sil_scores)]
        print(f"Optimal clusters for {ws}: {optimal_k}")

        # Final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        df_ws_sub['cluster'] = kmeans_final.fit_predict(features_scaled)

        # Store results
        results[ws] = {
            'data': df_ws_sub,
            'features_scaled': features_scaled,
            'clusters': df_ws_sub['cluster'],
            'kmeans': kmeans_final,
            'optimal_k': optimal_k,
        }

        # Optional: plot silhouette scores
        plt.figure(figsize=(6, 3))
        plt.plot(K_range, sil_scores, 'bo-')
        plt.title(f'Silhouette Scores for {ws}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()

    for ws in watersheds:
        print(f"\n--- Cluster summary for {ws} ---")
        df_sub = results[ws]['data']
        summary = df_sub.groupby('cluster')[['rmw100', 'speed100', 'vstore100', 'Compound_Area_sqkm']].mean().round(2)
        print(summary)

    for ws in watersheds:
        df_sub = results[ws]['data']
        features_scaled = results[ws]['features_scaled']

        pca = PCA(n_components=2)
        pca_comp = pca.fit_transform(features_scaled)

        df_sub.loc[:, 'pca1'] = pca_comp[:, 0]
        df_sub.loc[:, 'pca2'] = pca_comp[:, 1]

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x='pca1', y='pca2',
            hue='cluster',
            palette='tab10',
            size='Compound_Area_sqkm',
            sizes=(20, 200),
            alpha=0.7,
            data=df_sub
        )
        plt.title(f'PCA Clusters in Watershed {ws}')
        plt.show()

    for ws in watersheds:
        print(f"\nKruskal-Wallis tests for {ws}")
        df_sub = results[ws]['data']
        for col in ['rmw100', 'speed100', 'vstore100', 'Compound_Area_sqkm']:
            groups = [grp[col].values for _, grp in df_sub.groupby('cluster')]
            stat, p = kruskal(*groups)
            print(f"{col}: H={stat:.2f}, p={p:.4g}")