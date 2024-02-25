import sklearn
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def task2(table):

    df = pd.DataFrame(table)

    # Преобразуем DataFrame в NumPy array
    numpy_array = df.values

    # Разделим переменные на входные (X) и выходные (Y)
    X = numpy_array[:, :-1]  # Все столбцы, кроме последнего
    Y = numpy_array[:, -1]   # Последний столбец (class)

table = pd.read_csv("diabetes_data_uploadTRUE.csv")

def task3(table):
    df = pd.DataFrame(table)
    ages = df['Age']
    ages_scaled = preprocessing.scale(ages)
    print(ages)
    print("---------")
    print(ages_scaled)

def test_vib(table):
    df = pd.DataFrame(table)

    # Преобразуем DataFrame в NumPy array
    numpy_array = df.values

    # Разделим переменные на входные (X) и выходные (Y)
    X = numpy_array[:, :-1]  # Все столбцы, кроме последнего
    Y = numpy_array[:, -1]  # Последний столбец (class)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    print("Размерность X_train:", X_train.shape)
    print("Размерность X_test:", X_test.shape)
    print("Размерность Y_train:", Y_train.shape)
    print("Размерность Y_test:", Y_test.shape)

def transFrom_df(table):
    df = pd.DataFrame(table)
    # Преобразуйте столбец "Gender" в бинарные переменные
    df_encoded = pd.get_dummies(df, columns=['Gender','Polyuria','Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'], drop_first=True)
    # Сохраните X (все столбцы, кроме 'class') в файл X.csv
    df_encoded.drop(columns=['class']).to_csv('diabetes_data_upload.csv', index=False)

    # Сохраните 'class' в файл Y.csv
    df_encoded['class'].to_csv('Y.csv', index=False)

def train():
    table = pd.read_csv("X.csv")
    df = pd.DataFrame(table)
    # Преобразуем DataFrame в NumPy array
    numpy_arrayX = df.values
    y = pd.read_csv("Y.csv")
    df1 = pd.DataFrame(y)
    numpy_arrayY = df1.values
    # Разделим переменные на входные (X) и выходные (Y)
    X = numpy_arrayX
    Y = numpy_arrayY

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    # Создайте экземпляр модели
    model = LogisticRegression(random_state=0).fit(X_train,Y_train)

    # Обучите модель на обучающей выборке (X_train, Y_train)

    Y_pred = model.predict(X_test)

    print(Y_pred)
    print(accuracy_score(Y_test, Y_pred))
task2(table)
task3(table)
test_vib(table)
transFrom_df(table)
train()

