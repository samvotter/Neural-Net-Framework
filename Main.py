# Author: Sam Van Otterloo
# Title: Neural Net for Movie Recommendations

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

AND = []
with open("AND.csv") as andfile:
    rawAND = csv.reader(andfile)
    for row in rawAND:
        AND.append(m.Movie(row[0], float(row[1]), float(row[3]), "", float(row[2]), float(row[3])))

OR = []
with open("OR.csv") as orfile:
    rawOR = csv.reader(orfile)
    for row in rawOR:
        OR.append(m.Movie(row[0], float(row[1]), float(row[3]), "", float(row[2]), float(row[3])))


input_layer = n.NeuronCol(3)
output_layer = n.NeuronCol(1)

input_layer.connect(output_layer.neurons)

errors = []
avg = []
for i in range(0, 100):
    for movie in OR:
        gf.initialize(input_layer, movie)
        for neuron in input_layer.neurons:
            neuron.fire()
        for neuron in output_layer.neurons:
            neuron.compute_held()

        output_layer.neurons[0].compute_j(movie)

        # calculate output error
        output_layer.output_error(movie)

        # save the error
        errors.append(output_layer.neurons[0].j)

        for neuron in input_layer.neurons:
            neuron.adjust_weights()

        for neuron in output_layer.neurons:
            neuron.clean()

    avg.append(sum(errors)/len(errors))
    errors.clear()

# display results
plt.plot(avg)
plt.ylabel("Error Rate")
plt.xlabel("Iterations")
plt.title("Error Rate over Time")
plt.show()

'''
# construct the layers
input_layer = n.NeuronCol(3)
hidden_layer = n.NeuronCol(4)
h2 = n.NeuronCol(4)
output = n.NeuronCol(1)

# connect the layers
input_layer.connect(hidden_layer.neurons)
hidden_layer.connect(h2.neurons)
h2.connect(output.neurons)

# add bias neuron
input_layer.insert_bias(hidden_layer.neurons)
hidden_layer.insert_bias(h2.neurons)
h2.insert_bias(output.neurons)

avg = []
errors = []
guesses = []

x = 10000

# Introduction
print("Hello! \t This program is a Neural Net.\n")
print("It takes movie data from RottenTomatoes and IMDB and tries to tries guess how much *I* would like that movie.")
print("Because I am an obsessive dork ", end='')
print("I have already provided the Neural Net with my own spreadsheet of movie rankings.\n")
print("About the Data:")
print("\t 1) The data is 424 movies from 1940 to 2019 which I have seen and rated on a scale from 0 to 100.")
print("\t 2) While each Movie data includes the Title/Year/My Rating/Director/RottenTomatoes/IDMB")
print("\t\t only the Year/Rottentomatoes/IMDB scores are given to the neural net as inputs.")
print("\t 3) All inputs are represented as values between 0-100. To achieve this:")
print("\t\t Years: Year - (Current_Year-100)")
print("\t\t IMDB: IMDB*10")
print("\t For Example: \n\t\t Love Actually (2003) would be input as:")
print("\t\t\t Year: 2003 - (2019-100) \t 84")
print("\t\t\t RottenTomatoes:\t\t\t 63")
print("\t\t\t IMDB: 7.6*10  \t\t\t\t 76")

# train x times on . . .
for i in range(0, x):
    # the given data
    for movie in OR:
        # feed data
        gf.initialize(input_layer, movie)
        for neuron in input_layer.neurons:
            neuron.fire()
        input_layer.bias.fire()
        for neuron in hidden_layer.neurons:
            neuron.compute_held()
        for neuron in hidden_layer.neurons:
            neuron.fire()
        hidden_layer.bias.fire()
        for neuron in h2.neurons:
            neuron.compute_held()
        for neuron in h2.neurons:
            neuron.fire()
        h2.bias.fire()
        for neuron in output.neurons:
            neuron.compute_held()

        output.neurons[0].compute_j(movie)

        # calculate output error
        output.output_error(movie)

        # save the error
        errors.append(output.neurons[0].j)

        # Back Propogation
        for neuron in h2.neurons:
            neuron.compute_error()
        for neuron in hidden_layer.neurons:
            neuron.compute_error()

        # adjust weights
        for neuron in h2.neurons:
            neuron.adjust_weights()
        for neuron in hidden_layer.neurons:
            neuron.adjust_weights()
        for neuron in input_layer.neurons:
            neuron.adjust_weights()

        # clean the input garbage
        for neuron in h2.neurons:
            neuron.clean()
        for neuron in hidden_layer.neurons:
            neuron.clean()
        for neuron in output.neurons:
            neuron.clean()

    # calculate the average aggregate error
    avg.append(sum(errors)/len(errors))
    errors.clear()


# display results
plt.plot(avg)
plt.ylabel("Error Rate")
plt.xlabel("Iterations")
plt.title("Error Rate over Time")
plt.show()

print("\nIf you would like to know what the Neural Net *thinks* how I would rate your favorite movie, go ahead and")
print("input your own data. (You will need to look up the IMDB and RottenTomatoes scores)")
print("Don't manipulate the data at all. Just enter it as it appears. (IE: Year(1919-2019), RT(0-100), IMDB(0-10))")


choice = 'y'
while choice != 'N':
    title = input("Title:")
    year = int(input("Year:"))
    rt = int(input("RottenTomatoes:"))
    imdb = float(input("IMDB:"))

    imdb *= 10

    input_m = m.Movie(title, year, 0, "", rt, imdb)

    # feed data
    gf.initialize(input_layer, input_m)
    for neuron in input_layer.neurons:
        neuron.fire()
    input_layer.bias.fire()
    for neuron in hidden_layer.neurons:
        neuron.compute_held()
    for neuron in hidden_layer.neurons:
        neuron.fire()
    hidden_layer.bias.fire()
    for neuron in h2.neurons:
        neuron.compute_held()
    for neuron in h2.neurons:
        neuron.fire()
    h2.bias.fire()
    for neuron in output.neurons:
        neuron.compute_held()

    print("This Neural Net thinks I would rate", title, ":", output.neurons[0].held * 100, "/ 100")
    choice = input("Press 'N' to exit, or any other key try another movie.")

print("Thanks for playing! Bye.")
'''