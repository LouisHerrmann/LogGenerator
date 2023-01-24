import random
from replayer import *
from datetime import timedelta, datetime
import pandas as pd
import sys
import numpy as np


class MergedTraceGraph:
    def __init__(self, shared_act_dict):
        self.graph = nx.DiGraph()
        self.shared_act_dict = shared_act_dict
        self.last_id = 9999
        self.missing_objects = {}
        self.trace_paths = {}
        self.node_associated_objects = defaultdict(lambda: set())

    def get_new_id(self):
        self.last_id += 1
        return self.last_id

    def get_node_attributes_by_id(self, id):
        return self.graph.nodes.get(id)

    def get_node_related_objects(self, id):
        return self.graph.nodes.get(id)["related_objects"]

    def get_node_activity(self, id):
        return self.graph.nodes.get(id)["activity"]

    def get_nodes_for_activity(self, activity):
        related = []
        for node_id in self.graph.nodes:
            if self.get_node_activity(node_id) == activity:
                related.append(node_id)
        return related

    def get_node_associated_objects(self, node_id):
        return self.node_associated_objects[node_id]

    def get_first_node_with_uncovered_object_of_type(self, activity, object_type, last_merged_node):
        if len(list(self.shared_act_dict[activity])) == 1:
            # in case of non-shared activities, there is no point in searching for potential merge options
            return None
        if last_merged_node is None:
            search_space = self.graph.nodes
        else:
            search_space = []
            for succ in nx.dfs_successors(self.graph, last_merged_node).values():
                search_space += succ
        for node_id in search_space:
            if self.get_node_activity(node_id) == activity:
                if object_type in self.shared_act_dict[activity] and object_type not in self.get_node_related_objects(
                        node_id):
                    return node_id
        return None

    def add_object_to_node(self, node_id, object_type):
        current_objects = self.get_node_related_objects(node_id)
        current_objects.add(object_type)
        nx.set_node_attributes(self.graph, {node_id: {"related_objects": current_objects}})

    def get_nodes_with_missing_objects(self):
        return {node: objects for node, objects in self.missing_objects.items() if objects != set()}

    def get_node_with_missing_objects(self):
        for node, obj in self.missing_objects.items():
            if obj != set():
                return self.get_node_activity(node), object
        return None

    def get_activities_with_missing_objects_for_obj_type(self, object_type):
        return [self.get_node_activity(id) for id, objects in self.missing_objects.items() if object_type in objects]

    def get_first_missing_object_type(self):
        for s in self.missing_objects.values():
            if s != set():
                return list(s)[0]
        return None

    def add_trace(self, object_type, trace, trace_id, object_id):
        node_ids = []
        last_merged_node = None
        # add nodes
        for i in range(len(trace)):
            activity = trace[i]
            matching_node_id = self.get_first_node_with_uncovered_object_of_type(activity, object_type,
                                                                                 last_merged_node)
            if matching_node_id is None:
                node_id = self.get_new_id()
                required_objects = self.shared_act_dict[activity].difference([object_type])
                self.graph.add_node(node_id, activity=activity, related_objects={object_type})
                self.missing_objects[node_id] = required_objects
            else:
                node_id = matching_node_id
                last_merged_node = matching_node_id
                self.add_object_to_node(node_id, object_type)
                self.missing_objects[node_id] = self.missing_objects[node_id].difference([object_type])

            # keep track of node path for each trace and associate trace object ids related to a node
            node_ids.append(node_id)
            self.node_associated_objects[node_id].add(object_id)

            # add edge
            if i > 0:
                self.graph.add_edge(node_ids[-2], node_ids[-1], trace_id=trace_id)

        self.trace_paths[trace_id] = node_ids

    def unset_covered_traces(self, traces):
        for trace_id in self.trace_paths.keys():
            traces.get_trace_by_id(trace_id).covered = False

    def set_covered_traces(self, traces):
        for trace_id in self.trace_paths.keys():
            traces.get_trace_by_id(trace_id).covered = True

    def number_of_unmatched_events(self):
        return len([k for k, v in self.missing_objects.items() if v != set()])

    def convert_to_dataframe(self):
        topological_ordering = list(nx.topological_sort(self.graph))
        rows = []
        for node in topological_ordering:
            row = {"activity": self.get_node_activity(node)}
            associated_objects = list(self.get_node_associated_objects(node))
            for object_type in self.get_node_related_objects(node):
                row[object_type] = [obj for obj in associated_objects if obj.split("_")[0] == object_type]
            rows.append(row)

        return pd.DataFrame(rows)


def find_trace_with_most_matching_activities(activity_sequence, traces):
    # finds the trace that minimizes the number of open activities that still need matching after merge
    # i.e. if we have two traces t1, t2 with activities T1, M, T2 where M are the matching activities and T1, T2 the
    # remaining unmatched ones, we want to minimize |T1| + |T2|
    return []


class Traces:
    def __init__(self, traces_dict):
        self.traces_dict_by_id = {}
        self.traces_dict_by_object_type = defaultdict(lambda: set())
        self.last_id = 1000
        self.shared_act_dict = defaultdict(lambda: set())
        for object_type, traces in traces_dict.items():
            for trace in traces:
                self.traces_dict_by_id[self.last_id] = Trace(trace, object_type, self.last_id)
                self.traces_dict_by_object_type[object_type].add(self.last_id)
                self.last_id += 1
                for act in trace:
                    self.shared_act_dict[act].add(object_type)

    def get_object_types(self):
        return list(self.traces_dict_by_object_type.keys())

    def get_trace_by_id(self, id):
        return self.traces_dict_by_id[id]

    def get_traces_for_object_type(self, object_type):
        return self.traces_dict_by_object_type[object_type]

    def get_uncovered_traces(self):
        return [trace for trace in self.traces_dict_by_id.values() if not trace.covered]

    def get_traces_suitable_for_merging(self, object_type):
        possible_traces = self.get_uncovered_traces()
        if not possible_traces:
            possible_traces = list(self.traces_dict_by_id.values())
        possible_traces = [trace for trace in possible_traces if trace.object_type == object_type]
        if not possible_traces:
            # in case there are no traces for a given object type, we allow for others as well
            possible_traces = list(self.traces_dict_by_id.values())
        return possible_traces


class Trace:
    def __init__(self, sequence, object_type, id):
        self.sequence = sequence
        self.object_type = object_type
        self.id = id
        self.covered = False


class ObjectIdGenerator:
    # used to generate unique object identifiers for each object type
    def __init__(self, object_types):
        self.object_dict = {object_type: 1000 for object_type in object_types}

    def get_new_id(self, object_type):
        new_id = object_type + "_" + str(self.object_dict[object_type])
        self.object_dict[object_type] += 1
        return new_id


def combine_object_types(traces_dict, max_iterations, max_retries=5):
    traces = Traces(traces_dict)
    shared_act_dict = traces.shared_act_dict
    mergedGraphs = []
    object_id_generator = ObjectIdGenerator(traces.get_object_types())

    while traces.get_uncovered_traces():
        possible_choices = traces.get_uncovered_traces()
        chosen_trace = random.choice(possible_choices)
        chosen_trace.covered = True

        graph = MergedTraceGraph(shared_act_dict)
        new_object_id = object_id_generator.get_new_id(chosen_trace.object_type)
        graph.add_trace(chosen_trace.object_type, chosen_trace.sequence, chosen_trace.id, new_object_id)

        graph_per_reset = {}
        all_shared_activities_matched = False
        for j in range(max_retries):

            # build graph out of traces by randomly picking from the uncovered traces and matching shared activities
            for i in range(max_iterations):
                missing_object = graph.get_first_missing_object_type()
                if missing_object is None:
                    # all activities have been matched up
                    all_shared_activities_matched = True
                    break

                possible_choices = traces.get_traces_suitable_for_merging(missing_object)
                next_trace = random.choice(possible_choices)
                next_trace.covered = True
                new_object_id = object_id_generator.get_new_id(next_trace.object_type)
                graph.add_trace(next_trace.object_type, next_trace.sequence, next_trace.id, new_object_id)

            graph_per_reset[j] = graph

            if graph.get_first_missing_object_type() is None:
                all_shared_activities_matched = True
                break

            # in case we did not manage to match all shared activities, reset and try again
            # after max_retries times, we pick the one with the least number of violations
            graph.unset_covered_traces(traces)
            possible_choices = traces.get_uncovered_traces()
            chosen_trace = random.choice(possible_choices)
            chosen_trace.covered = True

            graph = MergedTraceGraph(shared_act_dict)
            new_object_id = object_id_generator.get_new_id(chosen_trace.object_type)
            graph.add_trace(chosen_trace.object_type, chosen_trace.sequence, chosen_trace.id, new_object_id)

        if all_shared_activities_matched:
            mergedGraphs.append(graph)
        else:
            graph.unset_covered_traces(traces)
            # pick the graph with the fewest number of unmatched shared activities
            graph = sorted([graph for graph in graph_per_reset.values()],
                           key=lambda g: len([k for k, v in g.missing_objects.items()
                                              if v != set()])
                           )[0]
            graph.set_covered_traces(traces)
            mergedGraphs.append(graph)

    return mergedGraphs


def convert_to_ocel(merged_graphs, start_date, min_time_stepsize, max_time_stepsize):
    # stepsizes in hours
    # convert each graph to a dataframe
    dataframes = [graph.convert_to_dataframe() for graph in merged_graphs]

    # randomly assign timestamps to events of each dataframe according to passed stepsize
    for index, df in enumerate(dataframes):
        date = start_date
        dates = []
        for i in range(len(df)):
            date += timedelta(hours=random.uniform(min_time_stepsize, max_time_stepsize))
            dates.append(date)
        df["end_time"] = dates

        # object ids are only unique within each dataframe, thus we postfix ids with index of df
        for ot in set(df.columns).difference(["end_time", "activity"]):
            df[ot] = df[ot].apply(lambda x: [obj + "_" + str(index) for obj in x] if x is not np.NaN else x)

    combined_df = pd.concat(dataframes).sort_values("end_time", ignore_index=True)

    # remove "BPMN_TASK(): ..." from activity names
    combined_df["activity"] = combined_df["activity"].apply(lambda x: x.split("BPMN_TASK(")[1].split(")")[0])

    # merge resulting dataframes and sort by timestamp
    return combined_df


def flatten_OCEL(dataframe):
    flattened_logs = {}
    object_types = set(dataframe.columns).difference(["activity", "end_time"])
    for ot in object_types:
        df_ot = dataframe.drop(columns=object_types.difference(ot)).dropna()
        # convert list of single object name into just object name
        df_ot[ot] = df_ot[ot].apply(lambda x: x[0])
        # change column names and order
        df_ot = df_ot[[ot, "activity", "end_time"]]
        df_ot.columns = ["traceid", "activity", "end_time"]
        flattened_logs[ot] = df_ot
    return flattened_logs


def get_parameters(parameter_path):
    param_file = open(parameter_path, "r")
    data = param_file.readline()
    param_file.close()
    data = data.replace("\n", "").split(";")
    params = []
    for i in range(len(data)):
        p = data[i]
        if i == 6:
            p = p.split(",")
            p = datetime(int(p[0]), int(p[1]), int(p[2]))
        else:
            p = int(p)
        params.append(p)
    return params


if __name__ == '__main__':
    # Replay parameters
    # max_trace_length: max supported trace length when simulating model
    # max_cycles:  max number of cycles until aborting when simulating model
    # ignore_self_loops: whether to ignore self loops when simulating model (0 or 1)
    # minimize_for_node_coverage: if this is false, we achieve path coverage under the parameter requirements
    #                                                                   (this can lead to longer replay times)

    # Merging parameters
    # max_iterations: max traces to combine when merging different object type's traces
    # max_retries: max number of tries we restart the merging of traces (in case, there still are unmatched shared
    #                                       activities after, we pick the one with the fewest unmatched activities)

    # Log generation parameters
    # start_date: start date of first event in log (YYYY, M, D)
    # min_time_stepsize: min time between events in hours
    # max_time_stepsize: max time between events in hours

    args = sys.argv
    if len(args) <= 1:
        input_path = "input/example.dot"
        output_path = "output"
        parameter_path = "parameters/config.txt"
    else:
        input_path = args[1]
        output_path = args[2]
        parameter_path = args[3]

    max_trace_length, max_cycles, ignore_self_loops, minimize_for_node_coverage, \
        max_iterations, max_retries, \
        start_date, min_time_stepsize, max_time_stepsize = get_parameters(parameter_path)

    graph = pydotplus.graph_from_dot_file(input_path).to_string()

    G, object_types = create_graph(graph)
    graphs = {}
    traces = {}
    for ot in object_types:
        # flatten input model for each object type
        graph = flatten_graph(G, ot)
        graphs[ot] = graph

        # replay each of the flattened logs and save replayed traces
        simulation = Simulation(graph,
                                max_trace_length=max_trace_length,
                                max_cycles=max_cycles,
                                minimize_for_node_coverage=minimize_for_node_coverage)
        simulation.start_simulation()
        traces[ot] = simulation.get_activity_sequence_representation(ignore_self_loops=ignore_self_loops)

    # combine all the replayed traces to partial order graphs covering all traces (merging on shared activities)
    merged_graphs = combine_object_types(traces,
                                         max_iterations=max_iterations,
                                         max_retries=max_retries)

    # convert the partial order graphs representing the merged traces to an OCEL format including timestamps
    dataframe = convert_to_ocel(merged_graphs, start_date=start_date,
                                min_time_stepsize=min_time_stepsize,
                                max_time_stepsize=max_time_stepsize)

    # check number of unmatched shared activities (goal: 0)
    unmatched_events = 0
    for graph in merged_graphs:
        unmatched_events += graph.number_of_unmatched_events()
    print("Number of unmatched events:", unmatched_events)

    print(len(dataframe), "events were generated")

    # flatten dataframe for each object type and save as csv
    flattened_logs = flatten_OCEL(dataframe)
    for ot, df in flattened_logs.items():
        df.to_csv(output_path + "/" + ot + ".csv", index=False)

    print(sum([len(df) for df in flattened_logs.values()]), "events after flattening")
