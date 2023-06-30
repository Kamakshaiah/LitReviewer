# https://www.analyticsvidhya.com/blog/2020/12/a-detailed-introduction-to-k-means-clustering-in-python/
# https://www.edlitera.com/en/blog/posts/pandas-add-rename-remove-columns
# https://realpython.com/k-means-clustering-python/
# https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score

data = pd.read_csv('D:\\Research\\papers\\finance\\iimv\\datasets\\cluster-anal\\audit-main-dataset.csv')


data = data.drop(columns=['Unnamed: 0'])

def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled

def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

def robust_scaling(df):
    # copy the dataframe
    df_robust = df.copy()
    # apply robust scaling
    for column in df_robust.columns:
        df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))
    return df_robust

##df_normal = robust_scaling(data)
##df_normal = df_normal.dropna()

# cluster analysis

# # DETERMINATION OF CLUSTERS - SCREEPLOT

##cluster_range = list(range(1, 11))
##inertias = []
##
##for c in cluster_range:
##    kmeans = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(data)
##    inertias.append(kmeans.inertia_)
##
##plt.figure()
##plt.plot(cluster_range, inertias, marker='o')
##plt.show()

# # DETERMINATION OF CLUSTERS - silhout method

##from sklearn.metrics import silhouette_samples, silhouette_score
##
##cluster_range = range(1, 11)
##results = []
##
##for c in cluster_range:
##    clusterer = KMeans(init='k-means++', n_clusters = c, n_init = 100, random_state=0).fit(data)
##    cluster_labels = clusterer.fit_predict(data)
##    silhouette_avg = silhouette_score(data, cluster_labels)
##    results.append([c, silhouette_avg])
##
##result = pd.DataFrame(results, columns = ['n_clusters', 'silhouette_score'])
##pivot_km = pd.pivot_table(result, index='n_clusters', values='silhouette_score')
##
##plt.figure()
##sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm.rocket_r)
##plt.tight_layout()

## CLUSTER ANALYSIS

kmeans = KMeans(init='k-means++', n_clusters = 5, n_init = 100, random_state=0).fit(data)
labels = pd.DataFrame(kmeans.labels_)
cluster_data = data.assign(Cluster=labels)
##sns.scatterplot(data=cluster_data)
##plt.show()

grouped_counts = cluster_data.groupby(['Cluster']).counts()

##X = data.values
##print(X.shape)
##kmeans = KMeans(n_clusters=5, random_state=0)
##fit = kmeans.fit(X)
##print(kmeans.cluster_centers_.shape)
##label = kmeans.predict(X)

##plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
##plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
##plt.scatter(X[y_predict== 2, 0], X[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
##plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
##plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
##plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
##plt.title('Clusters of customers')  
##plt.xlabel('Annual Income (k$)')  
##plt.ylabel('Spending Score (1-100)')  
##plt.legend()  
##plt.show()  

##plt.scatter(X[label == 0, 0], X[label == 0, 1], s=100, c='red', label='Cluster 1')
##plt.scatter(X[label == 1, 0], X[label == 1, 1], s=100, c='blue', label='Cluster 2')
##plt.scatter(X[label == 2, 0], X[label == 2, 1], s=100, c='green', label='Cluster 3')
##plt.scatter(X[label == 3, 0], X[label == 3, 1], s=100, c='cyan', label='Cluster 4')
##plt.scatter(X[label == 4, 0], X[label == 4, 1], s=100, c='yellow', label='Cluster 5')
##plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
##plt.legend()  
##plt.show()

##plt.scatter(X[label == 0, 1], X[label == 0, 2], s=100, c='red', label='Cluster 1')
##plt.scatter(X[label == 1, 1], X[label == 1, 2], s=100, c='blue', label='Cluster 2')
##plt.scatter(X[label == 2, 1], X[label == 2, 2], s=100, c='green', label='Cluster 3')
##plt.scatter(X[label == 3, 1], X[label == 3, 2], s=100, c='cyan', label='Cluster 4')
##plt.scatter(X[label == 4, 1], X[label == 4, 2], s=100, c='pink', label='Cluster 5')
##plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
##plt.legend()  
##plt.show()


##if __name__ == '__main__':
##    print(data.columns)
##    print(len(data.columns))
##    print(cluster_range)
##    print(df_normal.head())
    
