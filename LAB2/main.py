import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import LAB1.DataGenerator as dataGen


def draw_plt(data, indexes, title):
    plt.hist(data[indexes, 0], bins='auto', alpha=0.7)
    plt.hist(data[~indexes, 0], bins='auto', alpha=0.7)
    plt.title(title)
    plt.show()


def calculate_accurancy(pred, y):
    return sum(pred == y) / len(y)


def calculate_sensitivity_specificity(pred, y):
    # Рассчитаем количество TP, FN, TN, FP для выборки
    TP_test = sum((pred == 1) & (y == 1))
    FN_test = sum((pred == 0) & (y == 1))
    TN_test = sum((pred == 0) & (y == 0))
    FP_test = sum((pred == 1) & (y == 0))

    # Рассчитаем чувствительность и специфичность для выборки
    sensitivity = TP_test / (TP_test + FN_test)
    specificity = TN_test / (TN_test + FP_test)
    return sensitivity, specificity


def main():
    mode = int(input("Введите вариант генерации данных: "
                     "0 - линейно разделимые, 1 - нелинейнопересекаемые"))
    mu0, mu1, sigma0, sigma1 = [], [], [], []
    if mode == 0:
        mu0 = [0, 2, 3]
        mu1 = [8, 13, 6]
        sigma0 = [2, 1, 2]
        sigma1 = [1, 2, 1]
    elif mode == 1:
        mu0 = [0, 2, 3]
        mu1 = [3, 5, 1]
        sigma0 = [2, 1, 2]
        sigma1 = [1, 2, 1]

    mu = [mu0, mu1]
    sigma = [sigma0, sigma1]
    n = 1000 # число объектов класса

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

    title_test = "Результаты классификации, тестовая выборка"
    draw_plt(Pred_test_proba, ytest, title_test)

    Pred_train = clf.predict(xtrain)
    Pred_train_proba = clf.predict_proba(xtrain)

    title_train = "Результаты классификации, тренировочная выборка"
    draw_plt(Pred_train_proba, ytrain, title_train)

    # acc_train = clf.score(xtrain, ytrain)
    # acc_test = clf.score(xtest, ytest)

    acc_train = calculate_accurancy(Pred_train, ytrain)
    acc_test = calculate_accurancy(Pred_test, ytest)
    print("acc_train = ", acc_train, ", acc_test = ", acc_test)

    # Рассчитаем чувствительность и специфичность для тестовой выборки
    sensitivity_test, specificity_test = calculate_sensitivity_specificity(Pred_test, ytest)
    print("Sensitivity (Test):", sensitivity_test)
    print("Specificity (Test):", specificity_test)

    # Рассчитаем чувствительность и специфичность для тренировочной выборки
    sensitivity_train, specificity_train = calculate_sensitivity_specificity(Pred_train, ytrain)
    print("Sensitivity (Train):", sensitivity_train)
    print("Specificity (Train):", specificity_train)


if __name__ == '__main__':
    main()