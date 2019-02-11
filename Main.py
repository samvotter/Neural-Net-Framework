# libraries
import csv
import math

# modules
import Neuron as n
import Movie as m
import General_Functions as gf


movies = []
with open("Movies Masterlist.csv") as csvfile:
    rawdata = csv.reader(csvfile)
    header = 0
    for row in rawdata:
        if header == 0:
            header += 1
        else:
            movies.append(m.Movie(row[0], float(row[1]), float(row[2]), row[3], float(row[4]), float(row[5])))


# Construct the layers
input_layer = n.NeuronCol(3)
hidden_layer = n.NeuronCol(3)
output = n.NeuronCol(1)

# connect the layers
input_layer.connect(hidden_layer.neurons)
hidden_layer.connect(output.neurons)

# add bias neuron
input_layer.insert_bias(hidden_layer.neurons)
hidden_layer.insert_bias(output.neurons)

# feed data
gf.initialize(input_layer, movies[0])
for neuron in input_layer.neurons:
    print(neuron.held)
for neuron in input_layer.neurons:
    neuron.fire()
for neuron in hidden_layer.neurons:
    neuron.compute_held()
for neuron in hidden_layer.neurons:
    neuron.fire()
for neuron in output.neurons:
    neuron.compute_held()

print(output.neurons[0].held*100)
