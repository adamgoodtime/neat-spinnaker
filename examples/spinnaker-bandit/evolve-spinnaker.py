"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
from ast import literal_eval
import csv
import numpy as np
import time
import sys
from copy import deepcopy
from test_tf_xor import test_xor
from test_config import *

exec_thing = 'logic'
shared_probabilities = True
shape_fitness = True
spike_fitness = False
spike_weight = 0.1
noise_rate = 0
noise_weight = 0.01
fast_membrane = False
parse_conn = False
plasticity = False
delay = 25
parallel = False
repeat_best = 5
previous_best = False

# print ("reading from input")
# label = sys.argv[1]
# delay = float(sys.argv[2])
# print ("d", sys.argv[2])
# plasticity = bool(int(sys.argv[3]))
# print ("p", sys.argv[3])
# exec_thing = sys.argv[6]
# print ("e", sys.argv[6])
# print (delay)
# print (plasticity)
# print (exec_thing)
# '''


best_fitness = []
average_fitness = []
worst_fitness = []
repeated_fitness = []
best_score = []
average_score = []
worst_score = []

stats = None

weight = 0.1

iteration_count = 0

def read_fitnesses(config, max_fail_score=0):
    # fitnesses = []
    # file_name = 'fitnesses {}.csv'.format(config)
    # with open(file_name) as from_file:
    #     csvFile = csv.reader(from_file)
    #     for row in csvFile:
    #         metric = []
    #         for thing in row:
    #             metric.append(literal_eval(thing))
    #         fitnesses.append(metric)

    fitnesses = np.load('fitnesses {}.npy'.format(config))
    fitnesses = fitnesses.tolist()
    processed_fitness = []
    for fitness in fitnesses:
        processed_score = []
        for score in fitness:
            if score == 'fail':
                processed_score.append([max_fail_score, -10000000, -10000000])
            else:
                processed_score.append(score)
        processed_fitness.append(processed_score)
    return processed_fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def save_stats():
    global previous_best
    statistics = stats
    generation = len(statistics.most_fit_genomes)
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    # if len(best_fitness) > 0:
    #     previous_best = c
    #     np.save('best_agent {} {}.npy'.format(len(best_fitness)-1, config), c)
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())
    with open('NEAT stats {}.csv'.format(config), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(['Iteration: {}'.format(generation)])
        writer.writerow(['Best score'])
        writer.writerow(best_score)
        writer.writerow(['Repeated score'])
        writer.writerow(repeated_fitness)
        writer.writerow(['Average score'])
        writer.writerow(average_score)
        writer.writerow(['Worst score'])
        writer.writerow(worst_score)
        writer.writerow(['Best fitness'])
        writer.writerow(best_fitness)
        writer.writerow(['Average fitness'])
        writer.writerow(avg_fitness)
        writer.writerow(['Standard dev fitness'])
        writer.writerow(stdev_fitness)
        writer.writerow(['Current time'])
        writer.writerow([time.localtime()])
        writer.writerow(['Config'])
        writer.writerow([config])
        file.close()

def convert_genomes_to_matrix(genomes, input_size, output_size):
    inputs = [i - input_size for i in range(input_size)]
    outputs = [i for i in range(output_size)]
    agent_connections = []
    for agent in genomes:
        hidden_size = len(agent[1].nodes) - output_size
        in2rec = np.zeros([input_size, hidden_size])
        rec2rec = np.zeros([hidden_size, hidden_size])
        rec2out = np.zeros([hidden_size, output_size])
        in2out = np.zeros([input_size, output_size])
        hidden_neurons = []
        for conn in agent[1].connections:
            conn_params = agent[1].connections[conn]
            pre = conn_params.key[0]
            post = conn_params.key[1]
            if pre == post or (pre >= 0 and pre < output_size):
                continue
            if hidden_size == 0:
                in2out[pre + input_size][post] += conn_params.weight
            else:
                if post >= output_size:
                    if post not in hidden_neurons:
                        hidden_neurons.append(post)
                if pre >= output_size:
                    if pre not in hidden_neurons:
                        hidden_neurons.append(pre)
                if pre < 0:
                    if post < output_size:
                        in2out[pre+input_size][post] += conn_params.weight
                    else:
                        in2rec[pre+input_size][hidden_neurons.index(post)] += conn_params.weight
                else:
                    if post < output_size:
                        rec2out[hidden_neurons.index(pre)][post] += conn_params.weight
                    else:
                        rec2rec[hidden_neurons.index(pre)][hidden_neurons.index(post)] += conn_params.weight
        agent_connections.append([in2rec, rec2rec, rec2out, in2out, hidden_size])
    return agent_connections


def spinn_genomes(genomes, neat_config):
    global input_size, output_size, iteration_count
    input_size = neat_config.genome_config.num_inputs
    save_stats()
    agent_connections = convert_genomes_to_matrix(genomes, 2, 2)
    fitnesses = test_xor(agent_connections, 2, 2)
    # i = 0
    # print(config)
    # score_weighting = [1, 2]
    # shaped_fitnesses = [0 for i in range(len(fitnesses[0]))]
    # indexed_fitness = []
    # # labels the fitness with the agent id
    # for i in range(len(fitnesses)):
    #     new_indexes = []
    #     for j in range(len(fitnesses[i])):
    #         new_indexes.append([fitnesses[i][j], j])
    #     new_indexes.sort()
    #     indexed_fitness.append(new_indexes)
    # # ranks the fitnesses relative to each other
    # for metric, weight in zip(indexed_fitness, score_weighting):
    #     current_shape = 0
    #     for i in range(len(metric)):
    #         if i > 0:
    #             # if the fitness is the same as the previous agent give it the same score otherwise increase
    #             # the increment to the current rank
    #             if metric[i][0] != metric[i - 1][0]:
    #                 current_shape = i
    #         shaped_fitnesses[metric[i][1]] += current_shape * weight
    i=0
    for genome_id, genome in genomes:
        # genome.fitness = shaped_fitnesses[i]
        genome.fitness = fitnesses[i]
        i += 1


def run(config_file, SpiNNaker=True):
    global stats
    # Load configuration.
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(neat_config)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-55')
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    if SpiNNaker:
        winner = p.run(spinn_genomes, 1000)
    else:
        winner = p.run(eval_genomes, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(neat_config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-21')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-spinnaker')
    run(config_path)