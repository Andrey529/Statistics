from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import DataGenerator as dataGen
from sklearn.metrics import roc_auc_score

def main():
    mode = int(input("Введите вариант генерации данных: "
                     "0 - линейно разделимые, 1 - нелинейнопересекаемые\n"))
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
    n = 1000  # число объектов класса
    mu = [mu0, mu1]
    sigma = [sigma0, sigma1]
    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    # разделяем данные на 2 подвыборки
    trainCount = round(0.7 * n * 2)
    xtrain = x[0:trainCount]
    xtest = x[trainCount:n * 2 + 1]
    ytrain = y[0:trainCount]
    ytest = y[trainCount:n * 2 + 1]
    print("Tree")
    # Initialize Decision Tree Classifier
    tree_clf = DecisionTreeClassifier(random_state=0)

    # Fit the model
    tree_clf.fit(xtrain, ytrain)

    # Make predictions
    Pred_test = tree_clf.predict(xtest)
    Pred_train = tree_clf.predict(xtrain)

    # Evaluate accuracy
    acc_train = tree_clf.score(xtrain, ytrain)
    acc_test = tree_clf.score(xtest, ytest)
    print("Train accuracy  = ", acc_train)
    print("Test accuracy = ", acc_test)
    # Рассчитаем количество TP, FN, TN, FP для тестовой выборки
    TP_test = sum((Pred_test == 1) & (ytest == 1))
    FN_test = sum((Pred_test == 0) & (ytest == 1))
    TN_test = sum((Pred_test == 0) & (ytest == 0))
    FP_test = sum((Pred_test == 1) & (ytest == 0))
    # Чувствительность и специфичность для тестовой выборки
    sensitivity_test = TP_test / (TP_test + FN_test)
    specificity_test = TN_test / (TN_test + FP_test)

    print("Test sensitivity = ", sensitivity_test)
    print("Test specificity = ", specificity_test)

    # Рассчитаем количество TP, FN, TN, FP для тренировочной выборки
    TP_train = sum((Pred_train == 1) & (ytrain == 1))
    FN_train = sum((Pred_train == 0) & (ytrain == 1))
    TN_train = sum((Pred_train == 0) & (ytrain == 0))
    FP_train = sum((Pred_train == 1) & (ytrain == 0))

    # Рассчитаем чувствительность и специфичность для тренировочной выборки
    sensitivity_train = TP_train / (TP_train + FN_train)
    specificity_train = TN_train / (TN_train + FP_train)

    print("Train sensitivity = ", sensitivity_train)
    print("Train specificity = ", specificity_train)

    print("------------------------------------\n")

    print("Forest")
    # Initialize Decision Tree Classifier
    rf_clf = RandomForestClassifier(random_state=0)  # You can adjust hyperparameters here

    # Fit the model
    rf_clf.fit(xtrain, ytrain)

    # Make predictions
    Pred_test = rf_clf.predict(xtest)
    Pred_test_proba = rf_clf.predict_proba(xtest)
    Pred_train = tree_clf.predict(xtrain)

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
    print("Train accuracy = ", acc_train)
    print("Test accuracy = ", acc_test)

    # Получим вероятности принадлежности к классу 1
    y_score_tree = tree_clf.predict_proba(xtest)[:, 1]
    y_score_rf = rf_clf.predict_proba(xtest)[:, 1]

    # Рассчитаем ROC-кривые
    fpr_tree, tpr_tree, _ = roc_curve(ytest, y_score_tree)
    fpr_rf, tpr_rf, _ = roc_curve(ytest, y_score_rf)

    # Рассчитаем площадь под кривой (AUC)
    auc_tree = auc(fpr_tree, tpr_tree)
    auc_rf = auc(fpr_rf, tpr_rf)

    # Рассчитаем количество TP, FN, TN, FP для тестовой выборки
    TP_test = sum((Pred_test == 1) & (ytest == 1))
    FN_test = sum((Pred_test == 0) & (ytest == 1))
    TN_test = sum((Pred_test == 0) & (ytest == 0))
    FP_test = sum((Pred_test == 1) & (ytest == 0))

    # Чувствительность и специфичность для тестовой выборки
    sensitivity_test = TP_test / (TP_test + FN_test)
    specificity_test = TN_test / (TN_test + FP_test)

    print("Test sensitivity ", sensitivity_test)
    print("Test specificity ", specificity_test)
    print("Test accuracy", acc_test)
    print("Train accuracy ", acc_train)

    # Рассчитаем количество TP, FN, TN, FP для тренировочной выборки
    TP_train = sum((Pred_train == 1) & (ytrain == 1))
    FN_train = sum((Pred_train == 0) & (ytrain == 1))
    TN_train = sum((Pred_train == 0) & (ytrain == 0))
    FP_train = sum((Pred_train == 1) & (ytrain == 0))

    # Рассчитаем чувствительность и специфичность для тренировочной выборки
    sensitivity_train = TP_train / (TP_train + FN_train)
    specificity_train = TN_train / (TN_train + FP_train)

    print("Train sensitivity ", sensitivity_train)
    print("Train specificity ", specificity_train)

    print("------------------------------------\n")

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

    param_grid = {
        'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # примеры значений глубины дерева
    }

    tree_clf = DecisionTreeClassifier(random_state=0)
    grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(xtrain, ytrain)

    best_tree_params = grid_search.best_params_
    # Вывод оптимальных гиперпараметров
    print("Оптимальные гиперпараметры для дерева решений:", best_tree_params)

    # 2. Определение оптимального количества деревьев в случайном лесе
    best_auc = 0
    best_n_estimators = 0

    for n_estimators in range(1, 301, 10):
        forest_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest_clf.fit(xtrain, ytrain)
        Pred_test_proba = forest_clf.predict_proba(xtest)[:, 1]
        auc_score = roc_auc_score(ytest, Pred_test_proba)
        if auc_score > best_auc:
            best_auc = auc_score
            best_n_estimators = n_estimators

    # Вывод оптимального количества деревьев в случайном лесе
    print("Оптимальное количество деревьев в случайном лесе:", best_n_estimators)

if __name__ == "__main__":
    main()

