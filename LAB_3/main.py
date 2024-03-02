import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
import DataGenerator as dataGen
from sklearn.metrics import roc_auc_score

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
    tree_clf = DecisionTreeClassifier(random_state=0)

    # Fit the model
    tree_clf.fit(xtrain, ytrain)

    # Make predictions
    Pred_test = tree_clf.predict(xtest)
    Pred_test_proba = tree_clf.predict_proba(xtest)
    print("Pred_test = ", Pred_test, ", Pred_test_proba = ", Pred_test_proba)

    # Evaluate accuracy
    acc_train = tree_clf.score(xtrain, ytrain)
    acc_test = tree_clf.score(xtest, ytest)
    print("acc_train = ", acc_train, ", acc_test = ", acc_test)

    # Initialize Decision Tree Classifier
    rf_clf = RandomForestClassifier(random_state=0)  # You can adjust hyperparameters here

    # Fit the model
    rf_clf.fit(xtrain, ytrain)



    # Make predictions
    Pred_test = rf_clf.predict(xtest)
    Pred_test_proba = rf_clf.predict_proba(xtest)
    print("Pred_test = ", Pred_test, ", Pred_test_proba = ", Pred_test_proba)

    y_proba_test = Pred_test_proba
    plt.hist(y_proba_test[ytest, 1], bins='auto', alpha=0.5, color='black', label='Class 0')
    plt.hist(y_proba_test[~ytest, 1], bins='auto', alpha=0.5, color='red', label='Class 1')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Probabilities (Test Set)')
    plt.legend()
    plt.show()
    # Evaluate accuracy
    acc_train = rf_clf.score(xtrain, ytrain)
    acc_test = rf_clf.score(xtest, ytest)
    print("acc_train = ", acc_train, ", acc_test = ", acc_test)

    # Получим вероятности принадлежности к классу 1
    y_score_tree = tree_clf.predict_proba(xtest)[:, 1]
    y_score_rf = rf_clf.predict_proba(xtest)[:, 1]

    # Рассчитаем ROC-кривые
    fpr_tree, tpr_tree, _ = roc_curve(ytest, y_score_tree)
    fpr_rf, tpr_rf, _ = roc_curve(ytest, y_score_rf)

    # Рассчитаем площадь под кривой (AUC)
    auc_tree = auc(fpr_tree, tpr_tree)
    auc_rf = auc(fpr_rf, tpr_rf)

    # Выведем результаты
    print(f"AUC для дерева решений: {auc_tree:.2f}")
    print(f"AUC для случайного леса: {auc_rf:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tree, tpr_tree, label='Дерево решений (AUC = {:.2f})'.format(auc_tree), color='b')
    plt.plot(fpr_rf, tpr_rf, label='Случайный лес (AUC = {:.2f})'.format(auc_rf), color='g')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые для дерева решений и случайного леса')
    plt.legend(loc="lower right")
    plt.show()


    y_proba_train = rf_clf.predict_proba(xtrain)



    plt.hist(y_proba_train[ytrain, 1], bins=20, alpha=0.5, color='blue', label='Class 0')
    plt.hist(y_proba_train[~ytrain, 1], bins=20, alpha=0.5, color='green', label='Class 1')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Probabilities (Train Set)')
    plt.legend()
    plt.show()

    y_pred = (y_proba_test[:, 1] > 0.5).astype(int)  # Предположим, что порог threshold = 0.5

    # Вычисляем TP, FP, TN, FN
    TP = np.sum((y_pred == 1) & (ytest == 1))
    FP = np.sum((y_pred == 1) & (ytest == 0))
    TN = np.sum((y_pred == 0) & (ytest == 0))
    FN = np.sum((y_pred == 0) & (ytest == 1))

    # Вычисляем чувствительность и специфичность
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    print("Чувствительность (TPR):", TPR)
    print("Специфичность (TNR):", TNR)

if __name__ == "__main__":
    main()
