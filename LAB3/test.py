import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
# Загрузим данные (например, Iris dataset)
iris = load_iris()
X, y = iris.data, iris.target

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Инициализируем случайный лес
rf_clf = RandomForestClassifier(random_state=42)

# Обучим модель на обучающей выборке
rf_clf.fit(X_train, y_train)

# Получим вероятности принадлежности к каждому классу для тестовой выборки
y_proba_test = rf_clf.predict_proba(X_test)

# Построим гистограмму вероятностей для класса 0 (первого класса)
plt.figure(figsize=(8, 6))
plt.hist(y_proba_test[:, 0], bins=20, alpha=0.5, color='blue', label='Class 0')
plt.hist(y_proba_test[:, 1], bins=20, alpha=0.5, color='green', label='Class 1')
plt.hist(y_proba_test[:, 2], bins=20, alpha=0.5, color='red', label='Class 2')
plt.xlabel('Probability')
plt.ylabel('Count')
plt.title('Distribution of Probabilities (Test Set)')
plt.legend()
plt.show()
