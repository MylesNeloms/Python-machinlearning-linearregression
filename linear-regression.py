import tensorflow
import keras
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as pyplot
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2","G3", "studytime","failures","absences"]]


label = "G3"

x = np.array(data.drop([label], 1))
y = np.array(data[label])

"Saving best trained model// model with best accuracy"
best = 0
for j in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

if acc > best:
    best = acc
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)
"""using pickle to save and load best model"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

'''creating graph of output'''
p = 'G1'
pyplot.style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

