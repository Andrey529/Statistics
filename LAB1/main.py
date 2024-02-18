import matplotlib.pyplot as plt
import DataGenerator as dataGen


def main():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    col = len(mu0)
    n = 1000  # число объектов класса

    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    # разделяем данные на 2 подвыборки
    trainCount = round(0.7 * n * 2)  # не забываем округлить до целого
    xtrain = x[0:trainCount]
    xtest = x[trainCount:n * 2 + 1]
    ytrain = y[0:trainCount]
    ytest = y[trainCount:n * 2 + 1]

    # TODO(): подписать названия графиков и оси
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


if __name__ == "__main__":
    # main()

    num_points = 300
    x1, y1, x2, y2 = dataGen.nonlinear_dataset_5var(num_points)

    # Визуализация данных
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('График распределения двух классов случайных чисел')
    plt.grid(True)
    plt.show()
