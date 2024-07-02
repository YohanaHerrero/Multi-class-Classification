import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#I use the iris dataset: 3 types of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2), petal and sepal length
#The rows are the samples and the columns: Sepal Length, Sepal Width, Petal Length and Petal Width.
#I use the second two pairs
pair = [1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target
print('Classes: ', np.unique(y))

#plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")

#create the support vector classification model
model = SVC(kernel='linear', gamma=.5, probability=True)
#train model
model.fit(X,y)
#find accuracy on the training dataset
y_predict = model.predict(X)
accuracy_score(y, y_predict)

#Utility function: plots the different boundaries to carry out the classification
plot_colors = "ryb"
plot_step = 0.02
def decision_boundary(X,y,model,iris, two=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap = plt.cm.RdYlBu)
    
    if two:
        cs = plt.contourf(xx, yy, Z, cmap = plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], label = y, cmap = plt.cm.RdYlBu, s = 15)
        plt.show()
    else:
        set_={0,1,2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where(y == i)
            if np.any(idx):
                set_.remove(i)
                plt.scatter(X[idx, 0], X[idx, 1], label = y, cmap = plt.cm.RdYlBu, edgecolor = 'black', s=15)
        for  i in set_:
            idx = np.where(iris.target == i)
            plt.scatter(X[idx, 0], X[idx, 1], marker = 'x', color = 'black')
        plt.show()
        
#plot the boundaries for the classifications 
decision_boundary(X, y, model, iris)
