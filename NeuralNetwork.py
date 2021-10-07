import numpy as np


class NeuralNetwork:
    fitness = 0.0

    def __init__(self, nn_shape=None):
        # Структура сети для информации
        self.nn_shape = nn_shape
        # Нейроны
        self.neurons = []
        # Веса нейронных связей
        self.weights = []
        # Смещение нейрона (bias)
        self.biases = []
        if nn_shape != None:
            ## сделать throw exception, если len(nn_shape) == 1
            # инициализируем нейроны
            for layer in nn_shape:
                self.neurons.append(np.zeros(layer))

            # инициализируем смещения
            for i in range(1, len(nn_shape)):
                self.biases.append(np.random.uniform(low=-0.2, high=0.2, size=(nn_shape[i])))
            ## Иницииализируем связи
            for i in range(1, len(nn_shape)):
                self.weights.append(np.random.uniform(low=-0.5, high=0.5, size=(nn_shape[i], nn_shape[i - 1])))

    def forward_propagation(self, data):
        if len(data) == len(self.neurons[0]):
            self.neurons[0][:] = data[:]

            for i in range(1, len(self.neurons)):
                for j in range(len(self.neurons[i])):
                    self.neurons[i][j] = sigmoid(
                        np.dot(self.neurons[i - 1], self.weights[i - 1][j]) + self.biases[i - 1][j])
        return self.neurons[-1]

    # Гауссовская мутация
    def mutation(self):
        layer_num_out = np.random.randint(1, len(self.neurons))
        neur_out = np.random.randint(0, len(self.neurons[layer_num_out]))
        neur_in = np.random.randint(0, len(self.neurons[layer_num_out - 1]))
        self.weights[layer_num_out - 1][neur_out][neur_in] += np.random.normal(loc=0, scale=2)

    def __str__(self):
        return f'NN of shape {self.nn_shape}'

    def print_weights(self):
        print(self.weights)

    def print_bias(self):
        print(self.biases)

    def print_neurons(self):
        print(self.neurons)


def sigmoid(x):
    s = np.array(x)
    s = 1/(1+np.exp(-s))
    return s


