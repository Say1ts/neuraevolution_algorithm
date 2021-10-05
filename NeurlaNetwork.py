import numpy as np


class NeuralNetwork:
    def __init__(self, shape=None):
        self.shape = shape
        self.neurons = []
        self.weights = []
        self.biases = []

        if self.shape is not None:
            # инициализируем нейроны
            for layer in self.shape:
                self.neurons.append(np.zeros(layer))

            # инициализируем смещения
            for i in range(1, len(self.shape)):
                self.biases.append(np.random.uniform(low=-0.2, high=0.2, size=(self.shape[i])))
            # Инициализируем связи
            for i in range(1, len(self.shape)):
                self.weights.append(np.random.uniform(low=-0.5, high=0.5, size=(self.shape[i], self.shape[i - 1])))


    def forward_propagation(self, inputs):
        if len(inputs) == len(self.neurons[0]):
            self.neurons[0][:] = inputs[:]

        for i in range(1, len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j] = sigmoid(np.dot(self.neurons[i-1], self.weights[i - 1][j]) + self.biases[i - 1][j])
        return self.neurons[-1]

    # Гауссова мутация
    def mutation(self):
        layer_num_out = np.random.randint(1, len(self.neurons))
        neur_out = np.random.randint(0, len(self.neurons[layer_num_out]))
        neur_in = np.random.randint(0, len(self.neurons[layer_num_out - 1]))
        self.weights[layer_num_out - 1][neur_out][neur_in] += np.random.normal(loc=0, scale=2)

    def __str__(self):
        return f'NN of shape {self.shape}'

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
