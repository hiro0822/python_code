#データの準備、make_blobs関数で特徴量の少ないデータを用意する
rom sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=300,
                n_features=2,
                centers=3,
                random_state=0)
#ランダムデータを可視化する
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(X[:, 0],
                        X[:, 1],
                        c='blue',
                        marker='o',
                        s=50)
#K-means法をデータに適用する
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
km.fit(X)
km_predict = km.predict(X)
#クラスタ毎に色分けした散布図を描画する
plt.scatter(X[km_predict == 0, 0],#グラフのxの値
            X[km_predict == 0, 1],#グラフのyの値
            s=50,
            c= 'green',
            label='cluster1')
plt.scatter(X[km_predict == 1, 0],
            X[km_predict == 1, 1],
            s=50,
            c= 'purple',
            label='cluster2')

plt.scatter(X[km_predict == 2, 0],
            X[km_predict == 2, 1],
            s=50,
            c= 'red',
            label='cluster3')

plt.scatter(X[km_predict == 3, 0],
            X[km_predict == 3, 1],
            s=50,
            c= 'blue',
            label='cluster4')

plt.scatter(X[km_predict == 4, 0],
            X[km_predict == 4, 1],
            s=50,
            c= 'orange',
            label='cluster5')
#クラスターの重心は⭐︎マークを表示
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='yellow',
            label='centroid')

#エルボー法を用いて妥当なクラスター数を見るける
#各クラスター数毎のSSEを格納しておくための配列を定義
distances=[]
#for文を使って10回データを学習させ、その都度SSEを配列に追加する
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    distances.append(km.inertia_)
#グラフを描画
plt.plot(range(1,11), distances, marker='.')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Distance')
#グラフからクラスター数が３を超えたあたりから減少のペースが緩やかになってきているため、クラスター数を３でクラスタリングしてみる
km = KMeans(n_clusters=3)
km.fit(X)
km_predict = km.predict(X)
#グラフを描画
plt.scatter(X[km_predict == 0,  0],
                   X[km_predict == 0, 1],
                   s=50,
                   c='green',
                   label='cluster1')

plt.scatter(X[km_predict == 1,  0],
                   X[km_predict == 1, 1],
                   s=50,
                   c='purple',
                   label='cluster2')

plt.scatter(X[km_predict == 2,  0],
                   X[km_predict == 2, 1],
                   s=50,
                   c='red',
                   label='cluster3')

plt.scatter(km.cluster_centers_[:, 0],
                   km.cluster_centers_[:, 1],
                   s=250,
                   marker='*',
                   c='yellow',
                   label='centroid')


