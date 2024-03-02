import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
import DataGenerator as dataGen


def draw_0var():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    col = len(mu0)
    n = 1000  # число объектов класса

    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    # построение гистограмм распределения для всех признаков
    for i in range(0, col):
        _ = plt.hist(class0[:, i], bins='auto', alpha=0.7)  # параметр alpha позволяет задать прозрачность цвета
        _ = plt.hist(class1[:, i], bins='auto', alpha=0.7)
        plt.savefig('returns data/hist_' + str(i + 1) + '.png')  # сохранение изображения в файл

    plt.show()

    # построение одной скатеррограммы по выбранным признакам
    plt.scatter(class0[:, 0], class0[:, 2], marker=".", alpha=0.7)
    plt.scatter(class1[:, 0], class1[:, 2], marker=".", alpha=0.7)
    plt.savefig('returns data/scatter.png')
    plt.show()


def draw_data(get_dataset):
    x1, y1, x2, y2 = get_dataset

    # Визуализация данных
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('График распределения двух классов случайных чисел')
    plt.grid(True)
    plt.show()


def main():
    print("Введите вариант для генерации данных: ")
    var = int(input())
    if var == 0:
        draw_0var()
    else:
        num_points = 300
        if var == 2:
            draw_data(dataGen.nonlinear_dataset_2var(num_points))
        elif var == 5:
            draw_data(dataGen.nonlinear_dataset_5var(num_points))


if __name__ == "__main__":
    main()