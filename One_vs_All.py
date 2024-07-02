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

#I train three classifiers and place them in the list my_models. For each class I take the class samples I would like to 
#classify, and the rest will be labelled as a dummy class. I repeat the process for each class. For each classifier, I plot the 
#decision regions. The class I am interested in is in red, and the dummy class is in blue. Similarly, the class samples are 
#marked in blue, and the dummy samples are marked with a black x.

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

dummy_class = y.max() + 1
#list used for classifiers 
my_models = []
#iterate through each class
for class_ in np.unique(y):
    #select the index of our  class
    select = (y==class_)
    temp_y = np.zeros(y.shape)
    #class, we are trying to classify 
    temp_y[y==class_] = class_
    #set other samples  to a dummy class 
    temp_y[y!=class_] = dummy_class
    #Train model and add to list 
    model = SVC(kernel='linear', gamma=.5, probability=True)    
    my_models.append(model.fit(X, temp_y))
    #plot decision boundary 
    decision_boundary(X, temp_y, model, iris)

#For each sample I calculate the probability of belonging to each class, not including the dummy class
probability_array=np.zeros((X.shape[0],3))
for j, model in enumerate(my_models):
    real_class = np.where(np.array(model.classes_)!=3)[0]
    probability_array[:,j] = model.predict_proba(X)[:,real_class][:,0]

#function to plot the probability of belonging to each class; each column is the probability of belonging to a class and the 
#row number is the sample number
def plot_probability_array(X, probability_array):
    plot_array = np.zeros((X.shape[0], 30))
    col_start = 0
    ones = np.ones((X.shape[0], 30))
    for class_,col_end in enumerate([10, 20, 30]):
        plot_array[:,col_start:col_end] = np.repeat(probability_array[:, class_].reshape(-1, 1), 10, axis=1)
        col_start = col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()
    
#plot the probability of belonging to the class
plot_probability_array(X, probability_array)

#apply the argmax function to each sample to find the class
one_vs_all = np.argmax(probability_array, axis=1)
#calculate model accuracy
accuracy_score(y, one_vs_all)
