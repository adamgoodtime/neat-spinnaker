"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
from ast import literal_eval
import csv

# arms = [[0.9, 0.1], [0.1, 0.9]]
arms = [[0, 1], [1, 0]]
# arms = [[0, 1]]

number_of_arms = 2
number_of_epochs = 2
complimentary = True
shared_probabilities = True
grooming = 'cap'
reward_based = 0
spike_cap = 30000
spike_weight = 0.1
noise_rate = 100
noise_weight = 0.01
keys = ['fitness']

# UDP port to read spikes from
UDP_PORT1 = 17887
UDP_PORT2 = UDP_PORT1 + 1

number_of_trials = 105
duration_of_trial = 200
runtime = number_of_trials * duration_of_trial
try_attempts = 2
all_fails = 0
working_ends = 0
not_needed_ends = 0
# empty_pre_count = 0
empty_post_count = 0

weight_max = 1.0
weight_scale = 1.0
delay = 1

weight = 0.1

# exec_bandit = False
exec_bandit = True

config = 'a{}:{} -e{} - c{} - s{} - n{}-{} - g{} - r{}'.format(number_of_arms, arms[0], number_of_epochs, complimentary,
                                                                        shared_probabilities, noise_rate, noise_weight,
                                                                        grooming, reward_based)

input_size = 2
output_size = number_of_arms

def read_fitnesses(config):
    fitnesses = []
    file_name = 'fitnesses {}.csv'.format(config)
    with open(file_name) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            metric = []
            for thing in row:
                metric.append(literal_eval(thing))
            fitnesses.append(metric)
    return fitnesses

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def spinn_genomes(genomes, neat_config):
    globals()['pop'] = genomes
    globals()['arms'] = arms
    execfile("exec_bandit.py", globals())
    fitnesses = read_fitnesses(config)
    sorted_metrics = []
    combined_spikes = [[0, i] for i in range(len(genomes))]
    for i in range(len(fitnesses)):
        indexed_metric = []
        for j in range(len(fitnesses[i])):
            if fitnesses[i][j][0] == 'fail':
                indexed_metric.append([-10000000, j])
            else:
                indexed_metric.append([fitnesses[i][j][0], j])
            combined_spikes[j][0] -= fitnesses[i][j][1]
        indexed_metric.sort()
        sorted_metrics.append(indexed_metric)
    combined_spikes.sort()
    sorted_metrics.append(combined_spikes)

    if grooming != 'cap':
        combined_fitnesses = [0 for i in range(len(genomes))]
        for i in range(len(genomes)):
            for j in range(len(sorted_metrics)):
                combined_fitnesses[sorted_metrics[j][i][1]] += i
    else:
        combined_fitnesses = [0 for i in range(len(genomes))]
        for i in range(len(genomes)):
            for j in range(len(arms)):
                combined_fitnesses[sorted_metrics[j][i][1]] += sorted_metrics[j][i][0]
    i = 0
    for genome_id, genome in genomes:
        genome.fitness = combined_fitnesses[i]
        i += 1


def run(config_file, SpiNNaker=True):
    # Load configuration.
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(neat_config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    if SpiNNaker:
        winner = p.run(spinn_genomes, 300)
    else:
        winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-spinnaker')
    run(config_path)