import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
import DataGenerator as dataGen

def main():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    col = len(mu0)
    n = 1000  # число объектов класса

    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    # разделяем данные на 2 подвыборки
    trainCount = round(0.7 * n * 2)  # не забываем округлить до целого
    xtrain = x[0:trainCount]
    xtest = x[trainCount:n * 2 + 1]
    ytrain = y[0:trainCount]
    ytest = y[trainCount:n * 2 + 1]

    # Initialize Decision Tree Classifier
    clf = RandomForestClassifier(random_state=0)  # You can adjust hyperparameters here

    # Fit the model
    clf.fit(xtrain, ytrain)

    # Make predictions
    Pred_test = clf.predict(xtest)
    Pred_test_proba = clf.predict_proba(xtest)
    print("Pred_test = ", Pred_test, ", Pred_test_proba = ", Pred_test_proba)

    # Evaluate accuracy
    acc_train = clf.score(xtrain, ytrain)
    acc_test = clf.score(xtest, ytest)
    print("acc_train = ", acc_train, ", acc_test = ", acc_test)

if __name__ == "__main__":
    main()