import numpy as np
import pandas as pd
# Создание массивов случайных чисел

#1
from func import func

uniform_array = np.random.rand(23, 23)

normal_array = np.random.randn(23, 23)

#2
print("Равномерно распределенные")
print(uniform_array)
print("Нормально распределенные случайные числа")
print(normal_array)

#3
# Изучение атрибутов массива uniform_array
print("Атрибуты массива uniform_array:")
print("Количество осей (размерность):", uniform_array.ndim)
print("Размеры массива:", uniform_array.shape)
print("Общее количество элементов массива:", uniform_array.size)
print("Тип данных элементов массива:", uniform_array.dtype)
print("Размер в байтах каждого элемента массива:", uniform_array.itemsize)
print()


# Изучение атрибутов массива normal_array
print("Атрибуты массива normal_array:")
print("Количество осей (размерность):", normal_array.ndim)
print("Размеры массива:", normal_array.shape)
print("Общее количество элементов массива:", normal_array.size)
print("Тип данных элементов массива:", normal_array.dtype)
print("Размер в байтах каждого элемента массива:", normal_array.itemsize)


func(uniform_array)

table = pd.read_csv("diabetes_data_uploadTRUE.csv")


table['class'] = table['class'].map({'Positive': 1,'Negative': 0})
# Сохранение полученной таблицы в формате CSV
table.to_csv("table_numeric.csv", index=False)
