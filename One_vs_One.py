import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd

#I use the iris dataset: 3 types of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2), petal and sepal length
#The rows are the samples and the columns: Sepal Length, Sepal Width, Petal Length and Petal Width.
#I use the second two pairs
pair = [1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target
classes_=set(np.unique(y))
print('Classes: ', classes_)

#plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")

#number of classifiers needed
K=len(classes_)
print('We need ', K*(K-1)/2, ' classifiers.')

#Utility function: plots the different boundaries to carry out the classification
plot_colors = "ryb"
plot_step = 0.02
def decision_boundary(X, y, model, iris, two=None):
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

#train a two-class classifier on each pair of classes. I plot the different training points for each of the two classes.
pairs = []
left_overs = classes_.copy()
#list used for classifiers 
my_models = []
#iterate through each class
for class_ in classes_:
    #remove class we have seen before 
    left_overs.remove(class_)
    #the second class in the pair
    for second_class in left_overs:
        pairs.append(str(class_)+' and '+str(second_class))
        print("class {} vs class {} ".format(class_,second_class) )
        temp_y = np.zeros(y.shape)
        #find classes in pair 
        select = np.logical_or(y==class_, y==second_class)
        #train model 
        model = SVC(kernel='linear', gamma=.5, probability=True)  
        model.fit(X[select,:], y[select])
        my_models.append(model)
        #Plot decision boundary for each pair and corresponding Training samples. 
        decision_boundary(X[select,:], y[select], model, iris, two=True)

#each column is the output of a classifier for each pair of classes and the output is the prediction
majority_vote_array = np.zeros((X.shape[0],3))
majority_vote_dict = {}
for j, (model, pair) in enumerate(zip(my_models, pairs)):
    majority_vote_dict[pair] = model.predict(X)
    majority_vote_array[:,j] = model.predict(X)

pd.DataFrame(majority_vote_dict).head(10)

#perform a majority vote, that is, select the class with the most predictions. We repeat the process for each sample
one_vs_one = np.array([np.bincount(sample.astype(int)).argmax() for sample  in majority_vote_array]) 
#calculate the accuracy
accuracy_score(y, one_vs_one)
