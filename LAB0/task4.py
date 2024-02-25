import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def analyze_dataframe(df):
    # Информация об индексах
    print("Информация об индексах:")
    print(df.index)
    print("\n")

    # Информация о типах данных
    print("Информация о типах данных:")
    print(df.dtypes)
    print("\n")

    # Описательная статистика
    print("Описательная статистика:")
    print(df.describe())
    print("\n")

    # Первые 5 строк для первых 25 столбцов
    print("Первые 5 строк для первых 4 столбцов:")
    print(df[df.columns[:4]].head())

def yes_no_frames(df):
    # Создайте DataFrame, содержащий только строки со значением 'Yes' в колонке N+1
    df_yes = df.loc[df[df.columns[3]] == 'Yes']

    # Создайте DataFrame, содержащий только строки со значением 'No' в колонке N+1
    df_no = df.loc[df[df.columns[3]] == 'No']
    print("Yes")
    print(df_yes)
    print("No")
    print(df_no)

def sort_df(df):
    sorted_df = df.sort_values(by=[df.columns[2], df.columns[3], 'Age'])
    print(sorted_df)

def is_null_df(df):
    print("isna")
    print(df.isna())

    # Удалите строки, в которых есть хотя бы один пропуск
    df_no_missing = df.dropna()
    print(df_no_missing)

def make_gistogram(df):
    # Создайте DataFrame, содержащий только строки со значением 'Yes' в колонке N+1
    df_yes = df.loc[df[df.columns[3]] == 'Yes']

    # Создайте DataFrame, содержащий только строки со значением 'No' в колонке N+1
    df_no = df.loc[df[df.columns[3]] == 'No']

    plt.figure(figsize=(12, 6))

    # Гистограмма для Polyuria == 'Yes'
    plt.subplot(1, 2, 1)
    plt.hist(df_yes['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Age Distribution (Polyuria == Yes)')
    plt.xlabel('Age')
    plt.ylabel('Count')

    # Гистограмма для Polyuria == 'No'
    plt.subplot(1, 2, 2)
    plt.hist(df_no['Age'], bins=10, color='salmon', edgecolor='black')
    plt.title('Age Distribution (Polyuria == No)')
    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def make_boxplot(df):
    # Разделим данные на две группы по значению Polyuria
    yes_age = df[df['Polyuria'] == 'Yes']['Age']
    no_age = df[df['Polyuria'] == 'No']['Age']

    # Построим boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([yes_age, no_age], labels=['Yes', 'No'])
    plt.xlabel('Polyuria')
    plt.ylabel('Age')
    plt.title('Boxplot распределения возраста')
    plt.grid(True)
    plt.show()

def make_scatter_matrix(df):

    data = pd.read_csv("test_1and0.csv")
    pd.plotting.scatter_matrix(data.iloc[:,[0,2,3]], c= data["class"].replace(["1","0"],["blue","red"]))
    plt.show()

table = pd.read_csv("diabetes_data_uploadTRUE.csv")
analyze_dataframe(table)
yes_no_frames(table)
sort_df(table)
is_null_df(table)
make_gistogram(table)
make_boxplot(table)
make_scatter_matrix(table)
