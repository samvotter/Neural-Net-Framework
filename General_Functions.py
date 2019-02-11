import math
import statistics
import Neuron as n
import Movie as m


def sigmoid(number):
    value = 1/(1 + (math.exp(-number)))
    return value


def initialize(col, data):
    col.neurons[0].held = data.year - 1919
    col.neurons[1].held = data.rt
    col.neurons[2].held = data.imdb


def normalize_col(col):
    normalized = []

    average = sum(col) / len(col)

    for item in col:
        normalized.append((item - average) / statistics.stdev(col))

    return normalized

