import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#function to plot the probability of belonging to each class; each column represents the probability of belonging to a class and the row number is the sample number
def plot_probability_array(X, probability_array):
    plot_array = np.zeros((X.shape[0], 30))
    col_start = 0
    ones = np.ones((X.shape[0], 30))
    for class_, col_end in enumerate([10, 20, 30]):
        plot_array[:, col_start:col_end] = np.repeat(probability_array[:, class_].reshape(-1, 1), 10, axis=1)
        col_start = col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()

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

#logistic regression model
lr = LogisticRegression(random_state=0).fit(X, y)

#calculate probabilities
probability = lr.predict_proba(X)
#plot it
plot_probability_array(X, probability)

#apply the argmax function to the dataset to obtain predicted classes, which are the same of the indexes of max probability of x belonging to the 3 classes
softmax_prediction = np.argmax(probability, axis=1) #same as lr.predict(X) (directly done by sklearn though)

#compare to the sklearn predictions
y_predicted = lr.predict(X) #i can directly do this instead of calculating probabilities and then argmax
accuracy_score(y_predicted, softmax_prediction) 

#exactly the same

