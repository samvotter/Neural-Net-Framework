# libraries
import csv
import matplotlib.pyplot as plt

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
            movies.append(m.Movie(row[0], float(row[1]), float(row[2])/100, row[3], float(row[4]), float(row[5])))


# construct the layers
input_layer = n.NeuronCol(3)
hidden_layer = n.NeuronCol(3)
output = n.NeuronCol(1)

# connect the layers
input_layer.connect(hidden_layer.neurons)
hidden_layer.connect(output.neurons)

# add bias neuron
input_layer.insert_bias(hidden_layer.neurons)
hidden_layer.insert_bias(output.neurons)

avg = []
errors = []

# train x times on . . .
for i in range(0, 200):
    # the given data
    for movie in movies:
        # feed data
        gf.initialize(input_layer, movie)
        for neuron in input_layer.neurons:
            neuron.fire()
        for neuron in hidden_layer.neurons:
            neuron.compute_held()
        for neuron in hidden_layer.neurons:
            neuron.fire()
        for neuron in output.neurons:
            neuron.compute_held()

        # calculate output error
        output.output_error(movie)

        # save the error
        error = output.neurons[0].error

        for neuron in hidden_layer.neurons:
            neuron.compute_error()

        # adjust weights
        for neuron in hidden_layer.neurons:
            neuron.adjust_weights()
        for neuron in input_layer.neurons:
            neuron.adjust_weights()

        # clean the input garbage
        for neuron in hidden_layer.neurons:
            neuron.clean()
        for neuron in output.neurons:
            neuron.clean()

        # add the error to the error set
        errors.append(error)

    # calculate the average aggregate error
    avg.append(sum(errors)/len(errors))
    for item in errors:
        errors.remove(item)


# display results
plt.plot(avg)
plt.ylabel("Error Rate")
plt.xlabel("Iterations")
plt.show()
