import spynnaker8 as p
# from spynnaker.pyNN.connections. \
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
# from spinn_front_end_common.utilities.globals_variables import get_simulator
#
# import pylab
# from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
#     SpynnakerExternalDevicePluginManager as ex
import sys, os
import time
import socket
import numpy as np
from spinn_bandit.python_models.bandit import Bandit
import math
import itertools
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import csv
import threading
import pathos.multiprocessing
from spinn_front_end_common.utilities import globals_variables
from spinn_arm.python_models.arm import Arm
from ast import literal_eval

max_fail_score = -1000000


def get_scores(bandit_pop, simulator):
    b_vertex = bandit_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

def thread_bandit(pop, thread_arms, split=1, top=True):
    def helper(args):
        return test_pop(*args)

    step_size = len(pop) / split
    if step_size == 0:
        step_size = 1
    if isinstance(thread_arms[0], list):
        pop_threads = []
        all_configs = [[[pop[x:x + step_size], arm] for x in xrange(0, len(pop), step_size)] for arm in thread_arms]
        for arm in all_configs:
            for config in arm:
                pop_threads.append(config)
    else:
        pop_threads = [[pop[x:x + step_size], thread_arms] for x in xrange(0, len(pop), step_size)]
    pool = pathos.multiprocessing.Pool(processes=len(pop_threads))

    pool_result = pool.map(func=helper, iterable=pop_threads)

    pool.close()

    for i in range(len(pool_result)):
        new_split = 4
        if pool_result[i] == 'fail' and len(pop_threads[i][0]) > 1:
            print("splitting ", len(pop_threads[i][0]), " into ", new_split, " pieces")
            problem_arms = pop_threads[i][1]
            pool_result[i] = thread_bandit(pop_threads[i][0], problem_arms, new_split, top=False)

    agent_fitness = []
    for thread in pool_result:
        if isinstance(thread, list):
            for result in thread:
                agent_fitness.append(result)
        else:
            agent_fitness.append(thread)

    if isinstance(thread_arms[0], list) and top:
        copy_fitness = deepcopy(agent_fitness)
        agent_fitness = []
        for i in range(len(thread_arms)):
            arm_results = []
            for j in range(len(pop)):
                arm_results.append(copy_fitness[(i * len(pop)) + j])
            agent_fitness.append(arm_results)
    return agent_fitness

def connect_genes_to_fromlist(number_of_nodes, connections, nodes):
    i2i_ex = []
    i2i_in = []
    i2h_ex = []
    i2h_in = []
    i2o_ex = []
    i2o_in = []
    h2i_ex = []
    h2i_in = []
    h2h_ex = []
    h2h_in = []
    h2o_ex = []
    h2o_in = []
    o2i_ex = []
    o2i_in = []
    o2h_ex = []
    o2h_in = []
    o2o_ex = []
    o2o_in = []

    ex_or_in = {}
    translator = []
    i = 0
    for node in nodes:
        # translator.append([literal_eval(node), i])
        ex_or_in[node] = [nodes[node].activation, i]
        # print(node, ': ', ex_or_in[node])
        i += 1

    #individual: Tuples of (innov, from, to, weight, enabled)
    # if number_of_nodes > 4:
    #     print('hold it here')

    hidden_size = number_of_nodes - output_size

    for conn in connections:
        c = connections[conn]
        connect_weight = c.weight
        if c.enabled:
            # print('0:', c.key[0], '\t1:', c.key[1])
            if c.key[0] < 0:
                if c.key[1] < 0:
                    i2i_ex.append((c.key[0]+input_size, c.key[1]+input_size, connect_weight, delay))
                elif c.key[1] < output_size:
                    i2o_ex.append((c.key[0]+input_size, c.key[1], connect_weight, delay))
                elif ex_or_in[c.key[1]][1] < hidden_size + output_size:
                    i2h_ex.append((c.key[0]+input_size, ex_or_in[c.key[1]][1]-output_size, connect_weight, delay))
                else:
                    print("shit broke")
            elif c.key[0] < output_size:
                if c.key[1] < 0:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        o2i_ex.append((c.key[0], c.key[1]+input_size, connect_weight, delay))
                    else:
                        o2i_in.append((c.key[0], c.key[1]+input_size, connect_weight, delay))
                elif c.key[1] < output_size:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        o2o_ex.append((c.key[0], c.key[1], connect_weight, delay))
                    else:
                        o2o_in.append((c.key[0], c.key[1], connect_weight, delay))
                elif ex_or_in[c.key[1]][1] < hidden_size + output_size:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        o2h_ex.append((c.key[0], ex_or_in[c.key[1]][1]-output_size, connect_weight, delay))
                    else:
                        o2h_in.append((c.key[0], ex_or_in[c.key[1]][1]-output_size, connect_weight, delay))
                else:
                    print("shit broke")
            elif ex_or_in[c.key[0]][1] < hidden_size + output_size:
                if c.key[1] < 0:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        h2i_ex.append((ex_or_in[c.key[0]][1]-output_size, c.key[1]+input_size, connect_weight, delay))
                    else:
                        h2i_in.append((ex_or_in[c.key[0]][1]-output_size, c.key[1]+input_size, connect_weight, delay))
                elif c.key[1] < output_size:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        h2o_ex.append((ex_or_in[c.key[0]][1]-output_size, c.key[1], connect_weight, delay))
                    else:
                        h2o_in.append((ex_or_in[c.key[0]][1]-output_size, c.key[1], connect_weight, delay))
                elif ex_or_in[c.key[1]][1] < hidden_size + output_size:
                    if ex_or_in[c.key[0]][0] == 'excitatory':
                        h2h_ex.append((ex_or_in[c.key[0]][1]-output_size, ex_or_in[c.key[1]][1]-output_size, connect_weight, delay))
                    else:
                        h2h_in.append((ex_or_in[c.key[0]][1]-output_size, ex_or_in[c.key[1]][1]-output_size, connect_weight, delay))
                else:
                    print("shit broke")
            else:
                print("shit broke")


    return i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in

def connect_to_arms(pre_pop, from_list, arms, r_type):
    arm_conn_list = []
    for i in range(len(arms)):
        arm_conn_list.append([])
    for conn in from_list:
        arm_conn_list[conn[1]].append((conn[0], 0, conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(len(arms)):
        if len(arm_conn_list[i]) != 0:
            p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]), p.StaticSynapse(), receptor_type=r_type)

def connect_inputs(left, right, from_list, post_pop, r_type):
    input_conn_list = []
    for i in range(2):
        input_conn_list.append([])
    for conn in from_list:
        input_conn_list[conn[0]].append((0, conn[1], conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(2):
        if len(input_conn_list[i]) != 0:
            if i == 0:
                p.Projection(left, post_pop, p.FromListConnector(input_conn_list[i]),
                             receptor_type=r_type)
            else:
                p.Projection(right, post_pop, p.FromListConnector(input_conn_list[i]),
                             receptor_type=r_type)

def test_pop(pop, local_arms):#, noise_rate=50, noise_weight=1):
    #test the whole population and return scores
    global all_fails
    # global empty_pre_count
    global empty_post_count
    global not_needed_ends
    global working_ends
    print("start")
    on_rate = 15
    off_rate = 2
    # gen_stats(pop)
    # save_champion(pop)
    # tracker.print_diff()

    #Acquire all connection matrices and node types

    print(len(pop))
    # tracker.print_diff()
    #create the SpiNN nets
    scores = []
    spike_counts = []
    flagged_agents = []
    try_except = 0
    while try_except < try_attempts:
        print (config)
        left_input = []
        right_input = []
        input_count = -1
        # receive_on_pops = []
        hidden_node_pops = []
        hidden_count = -1
        hidden_marker = []
        output_pops = []
        # Setup pyNN simulation
        try:
            p.setup(timestep=1.0)
            p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
        except:
            print ("set up failed, trying again")
            try:
                p.setup(timestep=1.0)
                p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
            except:
                print ("set up failed, trying again for the last time")
                p.setup(timestep=1.0)
                p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
        for i in range(len(pop)):
            if i not in flagged_agents:
                number_of_nodes = len(pop[i][1].nodes) + len(local_arms)
                hidden_size = number_of_nodes - output_size

                [i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in] = \
                    connect_genes_to_fromlist(number_of_nodes, pop[i][1].connections, pop[i][1].nodes)
                # Create bandit population
                input_count += 1
                if local_arms[0] == 1:
                    left_input.append(p.Population(1, p.SpikeSourcePoisson(rate=on_rate), label='left on'))
                else:
                    left_input.append(p.Population(1, p.SpikeSourcePoisson(rate=off_rate), label='left off'))
                if local_arms[1] == 1:
                    right_input.append(p.Population(1, p.SpikeSourcePoisson(rate=on_rate), label='right on'))
                else:
                    right_input.append(p.Population(1, p.SpikeSourcePoisson(rate=off_rate), label='right off'))
                # choice = []
                # choice.append(p.Population(1, p.IF_cond_exp(), label='0_pop{}-{}'.format(input_count, i)))
                # choice.append(p.Population(1, p.IF_cond_exp(), label='0_pop{}-{}'.format(input_count, i)))
                output_pops.append(p.Population(2, p.IF_cond_exp(), label='output_pop{}-{}'.format(input_count, i)))
                output_pops[input_count].record('spikes')

                if hidden_size != 0:
                    hidden_node_pops.append(p.Population(hidden_size, p.IF_cond_exp(), label="hidden_pop {}".format(i)))
                    hidden_count += 1
                    hidden_marker.append(i)
                    if noise_rate != 0:
                        hidden_noise = p.Population(hidden_size, p.SpikeSourcePoisson(rate=noise_rate), label="hidden noise")
                        p.Projection(hidden_noise, hidden_node_pops[hidden_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    hidden_node_pops[hidden_count].record('spikes')
                # output_pops[i].record('spikes')

                if len(i2h_ex) != 0:
                    connect_inputs(left_input[input_count], right_input[input_count], i2h_ex,
                                   hidden_node_pops[hidden_count], 'excitatory')
                if len(i2o_ex) != 0:
                    connect_inputs(left_input[input_count], right_input[input_count], i2o_ex,
                                   output_pops[input_count], 'excitatory')
                if len(h2h_ex) != 0:
                    p.Projection(hidden_node_pops[hidden_count], hidden_node_pops[hidden_count], p.FromListConnector(h2h_ex),
                                 receptor_type='excitatory')
                if len(h2o_ex) != 0:
                        p.Projection(hidden_node_pops[hidden_count], output_pops[input_count], p.FromListConnector(h2o_ex),
                                     receptor_type='excitatory')
                if len(i2h_in) != 0:
                    connect_inputs(left_input[input_count], right_input[input_count], i2h_in,
                                   hidden_node_pops[hidden_count], 'inhibitory')
                if len(i2o_in) != 0:
                    connect_inputs(left_input[input_count], right_input[input_count], i2o_in,
                                   output_pops[input_count], 'inhibitory')
                if len(h2h_in) != 0:
                    p.Projection(hidden_node_pops[hidden_count], hidden_node_pops[hidden_count], p.FromListConnector(h2h_in),
                                 receptor_type='inhibitory')
                if len(h2o_in) != 0:
                        p.Projection(hidden_node_pops[hidden_count], output_pops[input_count], p.FromListConnector(h2o_in),
                                     receptor_type='inhibitory')

        print ("reached here 1")
        # tracker.print_diff()

        simulator = get_simulator()
        try:
            p.run(runtime)
            try_except = try_attempts
            print ("successful run")
            break
        except:
            traceback.print_exc()
            try:
                globals_variables.unset_simulator()
                working_ends += 1
            except:
                traceback.print_exc()
                not_needed_ends += 1
            all_fails += 1
            try_except += 1
            print ("\nfailed to run on attempt", try_except, ". total fails:", all_fails, "\n" \
                    "ends good/bad:", working_ends, "/", not_needed_ends)
            if try_except >= try_attempts:
                print ("calling it a failed population, splitting and rerunning")
                return 'fail'


    # hidden_count = 0
    # out_spike_count = [0 for i in range(len(pop))]
    # hid_spike_count = [0 for i in range(len(pop))]
    # print ("gathering spikes")
    # for i in range(len(pop)):
    #     if i not in flagged_agents:
    #         # spikes = output_pops[i].get_data('spikes').segments[0].spiketrains
    #         # for neuron in spikes:
    #         #     for spike in neuron:
    #         #         out_spike_count[i] += 1
    #         if i in hidden_marker:
    #             spikes = hidden_node_pops[hidden_count].get_data('spikes').segments[0].spiketrains
    #             hidden_count += 1
    #             for neuron in spikes:
    #                 for spike in neuron:
    #                     hid_spike_count[i] += 1
    #     else:
    #         print ("flagged ", i)
    #         out_spike_count[i] = 10000000
    #         hid_spike_count[i] = 10000000

    print ("reached here 2")
    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(pop))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(pop))]
    inhib_fail = 0
    for i in range(len(pop)):
        if i in flagged_agents:
            print ("worst score for the failure")
            fails += 1
            scores.append([0, 100000])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            spikes = output_pops[i - fails].get_data('spikes').segments[0].spiketrains
            on_spike = 0
            off_spike = 0
            j = 0
            for neuron in spikes:
                for spike in neuron:
                    if j == 1:
                        off_spike += 1
                    else:
                        on_spike += 1
                j += 1
            if (local_arms[0] == 1 and local_arms[1] == 1) or (local_arms[0] == 0 and local_arms[1] == 0):
                if reward_based == 1:
                    if off_spike > on_spike:
                        scores.append([1, on_spike + off_spike])
                    elif off_spike < on_spike:
                        scores.append([0, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
                else:
                    if on_spike:
                        scores.append([-1, on_spike + off_spike])
                    else:
                        scores.append([1, on_spike + off_spike])
            else:
                if reward_based == 1:
                    if off_spike < on_spike:
                        scores.append([1, on_spike + off_spike])
                    elif off_spike > on_spike:
                        scores.append([0, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
                else:
                    if on_spike:
                        scores.append([1, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        print ("\nfinished spikes")
        agent_fitness.append(scores[i][0])
    p.end()
    pop_fitnesses = []
    for i in range(len(pop)):
        pop_fitnesses.append(scores[i][0])#, out_spike_count[i] + hid_spike_count[i]])
        print (i, pop_fitnesses[i])

    return pop_fitnesses


def print_fitnesses(fitnesses):
    with open('fitnesses {}.csv'.format(config), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for fitness in fitnesses:
            writer.writerow(fitness)
        file.close()
    # with open('done {}.csv'.format(config), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     writer.writerow('', '')
    #     file.close()


if threading_tests:
    fitnesses = thread_bandit(pop, arms)
else:
    fitnesses = test_pop(pop, arms[0])

print_fitnesses(fitnesses)