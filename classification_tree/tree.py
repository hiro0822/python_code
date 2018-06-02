from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
#
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_axis, y_axis = np.meshgrid(np.arange(X_min, X_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
data_num = x_axis.shape[0] * x_axis.shape[1]
grid_points = np.concatenate((x_axis.reshape(data_num, 1), y_axis.reshape(data_num, 1)), axis=1)
#決定木モデルを読み込む
from sklearn.tree import DecisionTreeClassifier
#モデルへ適合する
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
#データを予測する
pred_label= tree.predict(grid_points)
pred_label = pred_label.reshape(x_axis.shape)
#予測結果のグラフ化
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
%matplotlib inline
markers = ('o','^','x')
colors = ('red', 'lightgreen' , 'cyan')
cmap = ListedColormap(colors)

for i, n in enumerate(np.unique(y)):
    plt.scatter(x=X_train[y_train == n, 0],
                        y=X_train[y_train == n, 1],
                        c= cmap(i),
                        marker=markers[i],
                        s = 70,
                        edgecolor='',
                        label=n)
plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='k',
                        marker='v',
                        label='test data')
plt.xlim(x_axis.min(), x_axis.max())
plt.ylim(y_axis.min(), y_axis.max())

plt.contourf(x_axis, y_axis, pred_label, alpha=0.3, cmap=cmap)

plt.legend(loc='lower right')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#分類過程の確認、grapthvizで分類過程を確認する
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot",
                feature_names=["petal length",petal width])
!dot -T png tree.dot -o tree.png
