import time
import csv
import pandas as pd

def task1(table):
    pandas_check(table)
    with open('diabetes_data_upload.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    no_pandas_check(data)

def pandas_check(table):
    start = time.time()
    average_age_all = table['Age'].mean()
    # Фильтрация данных для мужчин и женщин с использованием Pandas
    male_table = table[table['Gender'] == 'Male']
    female_table = table[table['Gender'] == 'Female']
    average_age_male = male_table['Age'].mean()
    average_age_female = female_table['Age'].mean()
    end = time.time()
    print("Средний возраст для всех людей (Pandas):", average_age_all)
    print("Средний возраст для женщин (Pandas):", average_age_female)
    print("Средний возраст для мужчин (Pandas):", average_age_male)
    print("Время Pandas:", end - start)

def no_pandas_check(data):
    start = time.time()
    average_age_male = 0
    male_count = 0
    average_age_female = 0
    female_count = 0
    for i in range(1, len(data)):
        if data[i][1] == "Male":
            average_age_male += int(data[i][0])
            male_count += 1
        else:
            average_age_female += int(data[i][0])
            female_count += 1
    average_age_all = (average_age_male+average_age_female)/(male_count+female_count)
    average_age_male /= male_count
    average_age_female /= female_count
    end = time.time()
    print("Средний возраст для всех людей (No Pandas):", average_age_all)
    print("Средний возраст для женщин (No Pandas):", average_age_female)
    print("Средний возраст для мужчин (No Pandas):", average_age_male)
    print("Время No Pandas:", end - start)

table = pd.read_csv("diabetes_data_upload.csv")
task1(table)

