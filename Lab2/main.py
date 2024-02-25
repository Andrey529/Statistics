import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import LAB1.DataGenerator as dataGen


def main():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    n = 1000  # число объектов класса

    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    # разделяем данные на 2 подвыборки
    trainCount = round(0.7 * n * 2)  # не забываем округлить до целого
    xtrain = x[0:trainCount]
    xtest = x[trainCount:n * 2 + 1]
    ytrain = y[0:trainCount]
    ytest = y[trainCount:n * 2 + 1]

    print("Train count = ", trainCount, ", test count = ", n * 2 - trainCount)

    Nvar = 2
    clf = LogisticRegression(random_state=Nvar, solver='saga').fit(xtrain, ytrain)

    Pred_test = clf.predict(xtest)
    Pred_test_proba = clf.predict_proba(xtest)

    plt.hist(Pred_test_proba[ytest, 0], bins='auto', alpha=0.7)
    plt.hist(Pred_test_proba[~ytest, 0], bins='auto', alpha=0.7)
    plt.title("Результаты классификации, тестовая выборка")
    plt.show()

    Pred_train = clf.predict(xtrain)
    Pred_train_proba = clf.predict_proba(xtrain)

    plt.hist(Pred_train_proba[ytrain, 0], bins='auto', alpha=0.7)
    plt.hist(Pred_train_proba[~ytrain, 0], bins='auto', alpha=0.7)
    plt.title("Результаты классификации, тренировочная выборка")
    plt.show()

    # acc_train = clf.score(xtrain, ytrain)
    # acc_test = clf.score(xtest, ytest)

    acc_train = sum(Pred_train == ytrain) / len(ytrain)
    acc_test = sum(Pred_test == ytest) / len(ytest)

    print("acc_train = ", acc_train, ", acc_test = ", acc_test)

    # Рассчитаем количество TP, FN, TN, FP для тестовой выборки
    TP_test = sum((Pred_test == 1) & (ytest == 1))
    FN_test = sum((Pred_test == 0) & (ytest == 1))
    TN_test = sum((Pred_test == 0) & (ytest == 0))
    FP_test = sum((Pred_test == 1) & (ytest == 0))

    # Рассчитаем чувствительность и специфичность для тестовой выборки
    sensitivity_test = TP_test / (TP_test + FN_test)
    specificity_test = TN_test / (TN_test + FP_test)

    print("Sensitivity (Test):", sensitivity_test)
    print("Specificity (Test):", specificity_test)

    # Рассчитаем количество TP, FN, TN, FP для тренировочной выборки
    TP_train = sum((Pred_train == 1) & (ytrain == 1))
    FN_train = sum((Pred_train == 0) & (ytrain == 1))
    TN_train = sum((Pred_train == 0) & (ytrain == 0))
    FP_train = sum((Pred_train == 1) & (ytrain == 0))

    # Рассчитаем чувствительность и специфичность для тренировочной выборки
    sensitivity_train = TP_train / (TP_train + FN_train)
    specificity_train = TN_train / (TN_train + FP_train)

    print("Sensitivity (Train):", sensitivity_train)
    print("Specificity (Train):", specificity_train)


if __name__ == '__main__':
    main()