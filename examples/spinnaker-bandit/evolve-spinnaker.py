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

print ("reading from input")
label = sys.arv[1]
delay = float(sys.argv[2])
print ("d", sys.argv[2])
plasticity = bool(int(sys.argv[3]))
print ("p", sys.argv[3])
exec_thing = sys.argv[6]
print ("e", sys.argv[6])
print (delay)
print (plasticity)
print (exec_thing)
# '''

threading_tests = True

try_attempts = 2
all_fails = 0
working_ends = 0
not_needed_ends = 0
# empty_pre_count = 0
empty_post_count = 0

'''remember to change inputs and outputs in the config as well'''

#arms params
arms_runtime = 20000
constant_input = 1
arms_stochastic = 1
arms_rate_on = 20
arms_rate_off = 5
random_arms = 0
arm1 = 1
arm2 = 0
arm3 = 0.1
arm_len = 1
arms = []
arms_reward = 1
for i in range(arm_len):
    arms.append([arm1, arm2])
    arms.append([arm2, arm1])
    # for arm in list(itertools.permutations([arm1, arm2, arm3])):
    #     arms.append(list(arm))
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0, 1], [1, 0]]
'''top_prob = 1
low_prob = 0
med_prob = 0.1
hii_prob = 0.2
arms = [[low_prob, med_prob, top_prob, hii_prob, med_prob, low_prob, med_prob, low_prob], [top_prob, low_prob, low_prob, med_prob, hii_prob, med_prob, low_prob, med_prob],
        [hii_prob, top_prob, med_prob, low_prob, low_prob, med_prob, med_prob, low_prob], [med_prob, low_prob, low_prob, top_prob, med_prob, hii_prob, low_prob, med_prob],
        [low_prob, low_prob, low_prob, med_prob, top_prob, med_prob, hii_prob, med_prob], [low_prob, med_prob, low_prob, med_prob, med_prob, top_prob, low_prob, hii_prob],
        [med_prob, low_prob, hii_prob, low_prob, med_prob, low_prob, top_prob, med_prob], [low_prob, hii_prob, med_prob, med_prob, low_prob, med_prob, low_prob, top_prob]]
# '''

#pendulum params
pendulum_runtime = 181000
double_pen_runtime = 60000
pendulum_delays = 1
max_fail_score = 0
no_v = False
encoding = 0
time_increment = 20
pole_length = 1
pole2_length = 0.1
pole_angle = [[0.1], [0.2], [-0.1], [-0.2]]
reward_based = 1
force_increments = 20
max_firing_rate = 1000
number_of_bins = 6
central = 1
bin_overlap = 2
tau_force = 0

#logic params
logic_runtime = 5000
score_delay = 5000
logic_stochastic = 0
logic_rate_on = 20
logic_rate_off = 5
truth_table = [0, 1, 1, 0]
# truth_table = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
input_sequence = []
segment = [0 for j in range(int(np.log2(len(truth_table))))]
input_sequence.append(segment)
for i in range(1, len(truth_table)):
    current_value = i
    segment = [0 for j in range(int(np.log2(len(truth_table))))]
    while current_value != 0:
        highest_power = int(np.log2(current_value))
        segment[highest_power] = 1
        current_value -= 2**highest_power
    input_sequence.append(segment)

#Recall params
recall_runtime = 60000
recall_rate_on = 50
recall_rate_off = 0
recall_pop_size = 1
prob_command = 1./6.
prob_in_change = 1./2.
time_period = 200
recall_stochastic = 1
recall_reward = 0
recall_parallel_runs = 2

#MNIST
max_freq = 5000
on_duration = 1000
off_duration = 1000
data_size = 200
mnist_parallel_runs = 2
mnist_runtime = data_size * (on_duration + off_duration)

#erbp params
erbp_runtime = 20
erbp_max_depth = [5, 100]

#breakout params
breakout_runtime = 181000
x_factor = 8
y_factor = 8
bricking = 0

inputs = 0
outputs = 0
test_data_set = []
config = ''
runtime = 0
stochastic = -1
rate_on = 0
rate_off = 0

if exec_thing == 'br':
    runtime = breakout_runtime
    inputs = (160 / x_factor) * (128 / y_factor)
    outputs = 2
    config = 'bout {}-{}-{} '.format(x_factor, y_factor, bricking)
    test_data_set = 'something'
    number_of_tests = 'something'
elif exec_thing == 'pen':
    runtime = pendulum_runtime
    constant_delays = pendulum_delays
    encoding = 1
    inputs = 4
    if encoding != 0:
        inputs *= number_of_bins
    if no_v:
        inputs /= 2
    outputs = 2
    config = 'pend-an{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'rank pen':
    runtime = pendulum_runtime
    constant_delays = pendulum_delays
    inputs = 4 * number_of_bins
    if no_v:
        config += "\b-no_v "
        inputs /= 2
    outputs = force_increments
    config = 'rank-pend-an{}-{}-F{}-R{}-B{}-O{}-E{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap, encoding)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'double pen':
    runtime = double_pen_runtime
    constant_delays = pendulum_delays
    inputs = 6 * number_of_bins
    if no_v:
        inputs /= 2
    outputs = force_increments
    config = 'double-pend-an{}-{}-pl{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), pole_length, pole2_length, force_increments, max_firing_rate, number_of_bins, bin_overlap)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'arms':
    stochastic = arms_stochastic
    rate_on = arms_rate_on
    rate_off = arms_rate_off
    if isinstance(arms[0], list):
        number_of_arms = len(arms[0])
    else:
        number_of_arms = len(arms)
    runtime = arms_runtime
    test_data_set = arms
    inputs = 2
    outputs = number_of_arms
    duration_of_trial = 200
    if random_arms:
        config = 'bandit-rand-{}-{} '.format(arms[0][0], len(arms))
    else:
        config = 'bandit-{}-{} '.format(arms[0][0], len(arms))
    if constant_input:
        if stochastic:
            config += 'stoc '
        config += 'on-{} off-{} r{} '.format(rate_on, rate_off, arms_reward)
    number_of_tests = len(arms)
elif exec_thing == 'logic':
    stochastic = logic_stochastic
    runtime = logic_runtime
    rate_on = logic_rate_on
    rate_off = logic_rate_off
    test_data_set = input_sequence
    number_of_tests = len(input_sequence)
    inputs = len(input_sequence[0])
    outputs = 2
    if stochastic:
        config = 'logic-stoc-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
    else:
        config = 'logic-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
    config += 'on-{} off-{} '.format(rate_on, rate_off)
elif exec_thing == 'recall':
    stochastic = recall_stochastic
    runtime = recall_runtime
    rate_on = recall_rate_on
    rate_off = recall_rate_off
    for j in range(recall_parallel_runs):
        test_data_set.append([j])
    number_of_tests = recall_parallel_runs
    inputs = 4 * recall_pop_size
    outputs = 2
    if stochastic:
        config = 'recall-stoc-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
    else:
        config = 'recall-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
elif exec_thing == 'mnist':
    runtime = mnist_runtime
    for j in range(mnist_parallel_runs):
        test_data_set.append([j])
    number_of_tests = mnist_parallel_runs
    inputs = 28*28
    outputs = 10
    config = 'mnist-freq-{}-on-{}-off-{}-size-{} '.format(max_freq, on_duration, off_duration, data_size)
elif exec_thing == 'erbp':
    maximum_depth = erbp_max_depth
    make_action = False
    runtime = erbp_runtime
    inputs = 0
    outputs = 0
    test_data_set = [[0], [1]]
    number_of_tests = len(test_data_set)
    config = 'erbp {} {} '.format(runtime, maximum_depth)
else:
    print("\nNot a correct test setting\n")
    raise Exception
if plasticity:
    config += 'pl '

if spike_fitness:
    if spike_fitness == 'out':
        config += 'out-spikes '
    else:
        config += 'spikes '
if shape_fitness:
    config += 'shape '
if noise_rate:
    config += 'noise {}-{} '.format(noise_rate, noise_weight)
if fast_membrane:
    config += 'fast_mem '
if parse_conn:
    config += 'parse_conn '
config += 'ish-{}'.format(label)
config += 'delay-{}'.format(delay)


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
    if len(best_fitness) > 0:
        previous_best = c
        np.save('best_agent {} {}.npy'.format(len(best_fitness)-1, config), c)
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

def spinn_genomes(genomes, neat_config):
    global input_size, output_size, iteration_count
    input_size = neat_config.genome_config.num_inputs
    save_stats()
    globals()['pop'] = deepcopy(genomes)
    if previous_best:
        for i in range(repeat_best):
            pop.append((0, previous_best))
    if exec_thing == 'xor':
        execfile("exec_xor.py", globals())
    else:
        execfile("exec_subprocess.py", globals())
    fitnesses = read_fitnesses(config)
    if spike_fitness:
        agent_spikes = []
        for k in range(len(pop)):
            spike_total = 0
            for j in range(len(test_data_set)):
                if isinstance(fitnesses[j][k], list):
                    spike_total -= fitnesses[j][k][1]
                    fitnesses[j][k] = fitnesses[j][k][0]
                else:
                    spike_total -= 1000000
            agent_spikes.append(spike_total)
        fitnesses.append(agent_spikes)
    sorted_metrics = []
    temp_scores = [0 for i in range(len(test_data_set))]
    if previous_best:
        for i in range(repeat_best):
            for j in range(len(test_data_set)):
                temp_scores[j] += float(fitnesses[j][len(genomes) + i][0])
        for j in range(len(test_data_set)):
            fitnesses[j] = fitnesses[j][0:len(genomes)]
            temp_scores[j] /= float(repeat_best)
        repeated_fitness.append(np.sum(temp_scores))
    combined_fitnesses = [0 for i in range(len(genomes))]
    combined_scores = [0 for i in range(len(genomes))]
    # combined_spikes = [[0, i] for i in range(len(genomes))]
    if exec_thing != 'xor':
        for i in range(len(fitnesses)):
            indexed_metric = []
            for j in range(len(fitnesses[i])):
                if fitnesses[i][j] == 'fail':
                    indexed_metric.append([-10000000, j])
                else:
                    if isinstance(fitnesses[i][j], list):
                        fitnesses[i][j] = fitnesses[i][j][0]
                    indexed_metric.append([fitnesses[i][j], j])
                # combined_spikes[j][0] -= fitnesses[i][j][1]
            indexed_metric.sort()
            sorted_metrics.append(indexed_metric)
        # combined_spikes.sort()
        # sorted_metrics.append(combined_spikes)

        # combined_fitnesses = [0 for i in range(len(genomes))]
        if shape_fitness:
            for j in range(len(sorted_metrics)):
                count = 0
                for i in range(len(genomes)):
                    if i > 0:
                        if sorted_metrics[j][i-1][0] != sorted_metrics[j][i][0]:
                            count = i
                    combined_fitnesses[sorted_metrics[j][i][1]] += count
                    if j < len(test_data_set):
                        combined_scores[sorted_metrics[j][i][1]] += sorted_metrics[j][i][0]
            best_index = combined_scores.index(max(combined_scores))
        else:
            for i in range(len(genomes)):
                for j in range(len(test_data_set)):
                    combined_fitnesses[sorted_metrics[j][i][1]] += sorted_metrics[j][i][0]
            best_index = combined_fitnesses.index(max(combined_fitnesses))
    else:
        for i in range(len(fitnesses)):
            for j in range(len(fitnesses[i])):
                if isinstance(fitnesses[i][j], list):
                    fitnesses[i][j] = fitnesses[i][j][0]
                combined_fitnesses[j] += fitnesses[i][j]
                # add spikes to fitness here somehow if you want
        best_index = combined_fitnesses.index(max(combined_fitnesses))
    i = 0
    for i in range(len(fitnesses[i])):
        print ("{:4} | ".format(i), end=" ")
        for j in range(len(fitnesses)):
            print (" {:6}".format(fitnesses[j][i]), end=" ")
        print (" \t {:6}".format(combined_fitnesses[i]))
    best_total = 0
    print("\nbest score is ", end=" ")
    for i in range(len(test_data_set)):
        print(fitnesses[i][best_index], end=" ")
        best_total += fitnesses[i][best_index]
    np.save('best agent score {} {}.npy'.format(iteration_count, config), genomes[best_index])
    iteration_count += 1
    best_score.append(best_total)
    print("best scores: ", best_score)
    print("repeated fitness: ", repeated_fitness)
    average_score.append(np.average(combined_scores))
    print("average scores: ", average_score)
    worst_score.append(min(combined_scores))
    print("worst scores: ", worst_score)
    print("\n", end=" ")
    save_stats()
    i = 0
    print(config)
    for genome_id, genome in genomes:
        genome.fitness = combined_fitnesses[i]
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