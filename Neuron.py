import General_Functions as gf
import random


class NeuronCol:

    def __init__(self, number):
        self.neurons = []
        for i in range(number):
            self.neurons.append(Neuron())

    def connect(self, col):
        for neuron in self.neurons:
            for neurons in col:
                neuron.weights[neurons] = random.randint(-20, 20)/100
                neuron.connections.append(neurons)

    def clean(self):
        for neuron in self.neurons:
            neuron.incoming.clear()

    def insert_bias(self, col):
        bias = Neuron()
        self.neurons.append(bias)
        for neurons in col:
            bias.weights[neurons] = 1
            bias.connections.append(neurons)
            bias.held = 1

    def output_error(self, movie):
        for neuron in self.neurons:
            neuron.error = (movie.my_rating - neuron.held) * neuron.held * (1 - neuron.held)


class Neuron:

    def __init__(self):
        self.incoming = []
        self.held = None
        self.weights = {}
        self.connections = []
        self.error = None

    def compute_held(self):
        total = 0
        for item in self.incoming:
            total += item
        adjusted = gf.sigmoid(total)
        self.held = adjusted

    def fire(self):
        for con in self.connections:
            con.incoming.append(self.held*self.weights[con])

    def computer_error(self, col):
        total = 0
        for con in self.connections:
            total += self.weights[con]*con.error
        total *= self.held * (1 - self.held)
        self.error = total
