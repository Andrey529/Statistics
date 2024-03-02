import numpy as np


def norm_dataset(mu, sigma, n):
    mu0 = mu[0]
    mu1 = mu[1]
    sigma0 = sigma[0]
    sigma1 = sigma[1]
    col = len(mu0)  # количество столбцов-признаков

    # инициализируем первый столбец (в Python нумерация от 0)
    class0 = np.random.normal(mu0[0], sigma0[0], [n, 1])
    class1 = np.random.normal(mu1[0], sigma1[0], [n, 1])

    for i in range(1, col):  # подумайте, почему нумерация с 1, а не с 0
        v0 = np.random.normal(mu0[i], sigma0[i], [n, 1])
        class0 = np.hstack((class0, v0))

        v1 = np.random.normal(mu1[i], sigma1[i], [n, 1])
        class1 = np.hstack((class1, v1))

    # создаем два массива для классификации объектов в сгенерированных массивах
    y1 = np.empty((n, 1), dtype=bool)  # пустой массив размерности Nx1
    y1.fill(1)  # заполнение пустого массива единицами
    y0 = np.empty((n, 1), dtype=bool)  # пустой массив размерности Nx1
    y0.fill(0)  # заполнение пустого массива нулями

    # конкатенируем два массива сгенерированных данных
    x = np.vstack((class0, class1))
    # конкатенируем два массива меток для сгенерированных данных
    y = np.vstack((y0, y1)).ravel()

    # перемешиваем данные
    rng = np.random.default_rng()
    arr = np.arange(2 * n)  # индексы для перемешивания
    rng.shuffle(arr)

    x = x[arr]
    y = y[arr]

    return x, y, class0, class1


def norm_definition(mu, sigma, n):
    return np.random.normal(mu, sigma, [n, 1])


def f1_2var(x):
    return 0.05 * (x ** 2) - x + 10


def f2_2var(x):
    return -0.05 * ((x - 8) ** 2) + x


def f1_5var(x):
    return x


def f2_5var(x):
    return 0.3 * x + 10


def f3_5var(x):
    return 3 * x - 31


def generate_random_points_along_function(num_points, func, x_range, noise_std_dev=0.0):
    # Генерация случайных значений для оси x в указанном диапазоне
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Вычисление значений для оси y, следующих функции func(x)
    y = func(x).copy()

    # Добавление случайного шума к значениям y, если указано
    if noise_std_dev > 0:
        for i in range(len(y)):
            noise = np.random.normal(0, noise_std_dev, 1)
            y[i] += noise[0]

    return x, y


def nonlinear_dataset_2var(num_points):
    x1_range = (6, 20)
    x2_range = (10, 26)
    noise_sigma = 0.5  # Стандартное отклонение для шума
    num_points = num_points // 2

    # Генерация случайных точек вдоль графика функции с использованием пользовательской функции
    x1, y1 = generate_random_points_along_function(num_points, f1_2var, x1_range, noise_sigma)
    x2, y2 = generate_random_points_along_function(num_points, f2_2var, x2_range, noise_sigma)
    return x1, y1, x2, y2


def nonlinear_dataset_5var(num_points):
    x1_range = (5, 16)
    x2_range = (8, 16)
    x3_range = (12, 16)
    noise_sigma = 0.5  # Стандартное отклонение для шума
    num_points = num_points // 3

    # Генерация случайных точек вдоль графика функции с использованием пользовательской функции
    x1, y1 = generate_random_points_along_function(num_points, f1_5var, x1_range, noise_sigma)
    x2_1, y2_1 = generate_random_points_along_function(num_points, f2_5var, x2_range, noise_sigma)
    x2_2, y2_2 = generate_random_points_along_function(num_points, f3_5var, x3_range, noise_sigma)

    x2 = np.vstack((x2_1, x2_2)).ravel()
    y2 = np.vstack((y2_1, y2_2)).ravel()

    return x1, y1, x2, y2
