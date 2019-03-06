import General_Functions as gf
import random

learning_rate = .8


class NeuronCol:

    def __init__(self, number):
        self.neurons = []
        self.bias = Neuron()
        for i in range(0, number):
            self.neurons.append(Neuron())

    def connect(self, col):
        for neuron in self.neurons:
            for neurons in col:
                neuron.weights[neurons] = random.randint(-20, 20)/100
                neuron.connections.append(neurons)

    def insert_bias(self, col):
        self.bias.held = int(1)
        for neurons in col:
            self.bias.weights[neurons] = 1
            self.bias.connections.append(neurons)

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
        self.j = None

    def compute_held(self):
        total = 0
        for item in self.incoming:
            total += item
        adjusted = gf.sigmoid(total)
        self.held = adjusted

    def fire(self):
        for con in self.connections:
            con.incoming.append(self.held*self.weights[con])

    def compute_error(self):
        total = 0
        for con in self.connections:
            total += self.weights[con]*con.error
        total *= self.held * (1 - self.held)
        self.error = total

    def adjust_weights(self):
        for con in self.connections:
            self.weights[con] += self.held * con.error * learning_rate

    def clean(self):
        self.incoming.clear()
        self.held = None

    def compute_j(self, movie):
        self.j = (movie.my_rating - self.held)**2



