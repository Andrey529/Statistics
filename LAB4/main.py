import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
import LAB1.DataGenerator as dataGen


# Активационная функция сигмоиды
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Производная сигмоиды
def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, x, y, n_neuro):
        self.input = x
        n_inp = self.input.shape[1]  # кол-во входов
        self.n_neuro = n_neuro  # число нейронов на главном слое
        # инициализация весов рандомными значениями
        self.weights1 = np.random.rand(n_inp, self.n_neuro)
        self.weights2 = np.random.rand(self.n_neuro, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.layer1 = 0
        self.layer2 = 0

    def feedforward(self):
        # выходы слоёв вычисляются по сигмоиде
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        # здесь происходит коррекция весов
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) *
                            sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T,
                            np.dot(
                                2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                self.weights2.T
                            ) * sigmoid_derivative(self.layer1))
        # обновляем веса
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        return self.weights1, self.weights2

    def train(self):
        # весь процесс обучения прост – высчитываем выход с помощью
        # прямого распространения, а после обновляем веса
        self.output = self.feedforward()
        return self.backprop()


def main():
    mu0 = [0, 2, 3]  # параметры выборки для генерации данных для нормального распределения
    mu1 = [3, 5, 1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    n = 1000  # число объектов класса
    mu = [mu0, mu1]
    sigma = [sigma0, sigma1]
    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)
    y = np.reshape(y, [2000, 1])

    n_neuro = 5  # число нейронов
    neural_network = NeuralNetwork(x, y, n_neuro)  # инициализируем сетку на наших данных
    count_epoch = 400
    losses = []  # список для хранения значений потерь
    accuracies = []  # список для хранения значений точности
    for i in range(count_epoch):
        print("for iteration # " + str(i) + "\n")
        # рассчет потерь как среднеквадратичное
        loss = np.mean(np.square(y - neural_network.feedforward()))
        print("Loss: \n" + str(loss))
        losses.append(loss)

        # получение предсказания
        pred = neural_network.feedforward()

        # подсчет точности
        count_matches = 0
        for j in range(len(pred)):
            if pred[j] >= 0.5 and y[j] == 1:
                count_matches += 1
        accuracy = count_matches / len(pred)
        accuracies.append(accuracy)
        print("Accuracy:", accuracy)

        # получение весов на каждой эпохе
        weights1, weights2 = neural_network.train()  # обучение сети
        print("weights1: ", weights1)
        print("weights2: ", weights2)

    # Построение графиков
    plt.figure(figsize=(10, 5))

    # График среднеквадратичной потери
    plt.subplot(1, 2, 1)
    plt.plot(range(1, count_epoch + 1), losses, label='Loss', color='red')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(range(1, count_epoch + 1), accuracies, label='Accuracy', color='blue')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
