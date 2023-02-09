import numpy as np
import random
from collections import Counter
from enum import Enum
from copy import deepcopy


class Dist(Enum):
    UNIFORM = 0
    BINOMIAL = 1
    GEOMETRIC = 2


def determine_graphs_for_sampling(merged_graphs):
    perfectly_matching_graphs = [graph for graph in merged_graphs if graph.number_of_unmatched_events() == 0]
    imperfectly_matching_graphs = [graph for graph in merged_graphs if graph not in perfectly_matching_graphs]
    if len(perfectly_matching_graphs) >= len(imperfectly_matching_graphs):
        # if there are more perfectly matching graphs, we just use these for the duplication
        return perfectly_matching_graphs, imperfectly_matching_graphs
    elif len(imperfectly_matching_graphs) > 3:
        # if there aren't enough perfectly matching graphs, we use the better half of the rest for the duplication
        s = sorted(imperfectly_matching_graphs,
                   key=lambda x: x.number_of_unmatched_events())
        return perfectly_matching_graphs + s[:int(len(s) / 2)+1], s[int(len(s) / 2)+1:]
    return perfectly_matching_graphs + imperfectly_matching_graphs, []


def filter_out_duplicates(merged_graphs):
    # filter out duplicate merged graphs that so that we can more reliably sample the data to create the right prob dist
    all_trace_paths = [graph.trace_paths for graph in merged_graphs]
    duplicates = []
    for i in range(len(all_trace_paths) - 1):
        if all_trace_paths[i] in all_trace_paths[i + 1:]:
            duplicates.append(i)
    return [merged_graphs[i] for i in range(len(merged_graphs)) if i not in duplicates]


def create_distribution(input_list, output_length, median_traces_per_graph, dist=Dist.GEOMETRIC, p=0.6):
    n = len(input_list)
    num_to_generate = max([int((output_length - n) / median_traces_per_graph), 0])
    samples = []

    if dist == Dist.UNIFORM:
        samples = random.choices(range(n), k=num_to_generate)
    elif dist == Dist.BINOMIAL:
        samples = [i - 1 for i in np.random.binomial(n, p, num_to_generate)]
    else:
        while len(samples) < num_to_generate:
            rand = np.random.geometric(p=p) - 1
            if rand < n:
                samples.append(rand)

    # translate index samples into actual items in input_list
    counts = Counter(samples)
    output = []
    for k, v in counts.items():
        for i in range(v):
            output.append(deepcopy(input_list[k]))
    # append input_list to output to ensure we cover all traces
    return output + input_list, samples
