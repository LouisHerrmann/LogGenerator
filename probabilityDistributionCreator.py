import pandas as pd
import numpy as np
import random
from collections import Counter
from enum import Enum
from copy import deepcopy


class Dist(Enum):
    UNIFORM = 0
    BINOMIAL = 1
    GEOMETRIC = 2


def filter_out_duplicates(merged_graphs):
    # filter out duplicate merged graphs that so that we can more reliably sample the data to create the right prob dist
    all_trace_paths = [graph.trace_paths for graph in merged_graphs]
    duplicates = []
    for i in range(len(all_trace_paths)-1):
        if all_trace_paths[i] in all_trace_paths[i+1:]:
            duplicates.append(i)
    return [merged_graphs[i] for i in range(len(merged_graphs)) if i not in duplicates]


def create_distribution(input_list, output_length, dist=Dist.GEOMETRIC, p=0.6, min_traces=0,
                        min_traces_per_obj_type=0, min_events_per_obj_type=0):
    n = len(input_list)
    num_to_generate = max([output_length - n, 0])
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
    return output + input_list
