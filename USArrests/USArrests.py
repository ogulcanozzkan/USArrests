## IMPORT
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


## DATA
df = pd.read_csv("USArrests.csv",index_col=0)
df.head()
df.isnull().sum()
df.info()
df.describe().T
df.hist(figsize=(10,10))
plt.show(block=True)


# K MEANS
kmeans = KMeans(n_clusters=4)
k_fit = kmeans.fit(df)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_

k_means = KMeans(n_clusters=2).fit(df)
kumeler = k_means.labels_
print(kumeler)

plt.scatter(df.iloc[:,0],df.iloc[:1],c=kumeler,s=50,cmap="viridis")
plt.scatter(merkezler[:,0],merkezler[:,0],c="black",s=200,alpha=0.5)
plt.show(block=True)

merkezler = k_means.cluster_centers_
print(merkezler)
plt.scatter(merkezler[:,0],merkezler[:,0],c="black",s=200,alpha=0.5)

ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

print(ssd)
plt.plot(K,ssd,"bx-")
plt.xlabel("farklı k değerlerine karşı uzaklık artık toplamları")
plt.title("Optimum küme sayısı için Elbow yöntemi")
plt.show(block=True)

kmeans = KMeans()
visu = KElbowVisualizer(kmeans,k=(2,20))
visu.fit(df)
visu.poof()

kmeans = KMeans(n_clusters=4).fit(df)
kumeler = kmeans.labels_
pd.DataFrame({"Eyaletler":df.index,"kumeler":kumeler})
df["Kume_no"]=kumeler
print(df)


# HIYERARŞIK KUMELEME
hc_complete = linkage(df,"complete")
hc_average = linkage(df,"average")

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
plt.show(block=True)
dendrogram(hc_complete,
           truncate_mode = "lastp",
           p=4,
           show_contracted = True,
           leaf_font_size=10);


# PCA
df1 = pd.read_csv("Hitters.csv")
df.dropna(inplace=True)
df1 = df._get_numeric_data()
df1.head()

from sklearn.preprocessing import StandardScaler
df1 = StandardScaler().fit_transform(df)
df1[0:5,0:5]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df1)
bilesen_df = pd.DataFrame(data=pca_fit,columns=["Birinci Bileşen","İkinci Bileşen"])
print(bilesen_df)

pca.explained_variance_ratio_
pca.components_

pca = PCA().fit(df1)
plt.plot(np.cumsum(pca.explained_variance_ratio_));
plt.xlabel("Bileşen sayısı")
plt.ylabel("Kümülatif Varyans")
plt.show(block=True)

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df1)
pca.explained_variance_ratio_

