#データセットの準備
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
X = X[0:100]
y = iris.target[0:100]
#データを訓練用とテスト用に分ける
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)
#SVMモデルを読み込む
from sklearn.svm import SVC
svm = SVC(kernel='linear')
#モデルへ適合させる
svm.fit(X_train, y_train)
#予測
from sklearn.metrics import accuracy_score
y_predicted = svm.predict(X_test)
#散布図で確認
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(X[:50,0], X[:50,1], color='blue',marker='o',label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='red',marker='o',label='versicolor')
plt.legend()

X_min, X_max = X[:, 0].min( ) - 1,  X[:, 0].max( ) + 1
y_min, y_max = X[:, 1].min( ) - 1,  X[:, 1].max( ) + 1

x_axis, y_axis = np.meshgrid(np.arange(X_min, X_max, 0.02),
                                                    np.arange(y_min, y_max, 0.02))
data_num = x_axis.shape[0] *x_axis.shape[1]
grid_points = np.concatenate((x_axis.reshape(data_num, 1),y_axis.reshape(data_num, 1)), axis=1)
class_labels = svm.predict(grid_points)
class_labels = class_labels.reshape(x_axis.shape)
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
%matplotlib inline

markers=('o', '^')
colors = ('red', 'lightgreen')
cmap = ListedColormap(colors)
labels = ('serosa', 'versicolor')

for i, n in enumerate(np.unique(y)):
        plt.scatter(x=X_train[y_train == n, 0],
                            y=X_train[y_train == n, 1],
                               c=cmap(i),
                               marker=markers[i],
                               s=70,
                           edgecolors='',
                           label=labels[i])
plt.legend(loc='lower right')

plt.xlim(x_axis.min(), x_axis.max())
plt.ylim(y_axis.min(), y_axis.max())

plt.contourf(x_axis,y_axis, class_labels, alpha=0.3, cmap=cmap)
#非線形のデータを読み込む
from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
#データを訓練用とテスト用に分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from matplotlib.colors import ListedColormap
markers=('o', '^')
colors = ('red', 'lightgreen')

cmap = ListedColormap(colors)
for i, n in enumerate(np.unique(y)):
    plt.scatter(x=X_train[y_train == n, 0],
                           y=X_train[y_train == n, 1],
                           c=cmap(i),
                           marker=markers[i],
                           s=70,
                           edgecolors='',
                           label=n)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#線形SVMでの正解率
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)
y_predicted = svm.predict(X_test)
print('線形SVMの正答率', accuracy_score(y_test, y_predicted))
#カーネルSVMでの正解率
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_predicted = svm.predict(X_test)
print('カーネルSVMの正答率', accuracy_score(y_test, y_predicted))
#カーネルSVMによるサンプルデータセットの分類を領域図にする
X_min, X_max = X[:, 0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
x_axis, y_axis = np.meshgrid(np.arange(X_min, X_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
grid_points = np.array([x_axis.ravel(), y_axis.ravel()])
grid_points =grid_points.T
class_labels = svm.predict(grid_points)
class_labels = class_labels.reshape(x_axis.shape)

from matplotlib.colors import ListedColormap
markers = ('o','^')
colors = ('red', 'lightgreen')
cmap = ListedColormap(colors)
labels = ('setona', 'versicolor')
for i, n in enumerate(np.unique(y)):
    plt.scatter(x=X_train[y_train == n, 0],
                       y=X_train[y_train == n, 1],
                       c=cmap(i),
                       marker=markers[i],
                       s=70,
                       edgecolors='',
                       label=labels[i])
plt.legend(loc='lower right')
plt.xlim(x_axis.min(), x_axis.max())
plt.xlim(y_axis.min(), y_axis.max())
plt.contourf(x_axis, y_axis, class_labels, alpha=0.3, cmap=cmap)
plt.show()
