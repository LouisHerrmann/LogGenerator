import random
from collections import defaultdict
import pandas as pd

import networkx as nx

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
                related.add(node_id)
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
                if object_type in self.shared_act_dict[activity] and object_type not in self.get_node_related_objects(node_id):
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
            matching_node_id = self.get_first_node_with_uncovered_object_of_type(activity, object_type, last_merged_node)
            if matching_node_id is None:
                node_id = self.get_new_id()
                required_objects = self.shared_act_dict[activity].difference([object_type])
                self.graph.add_node(node_id, activity=activity, related_objects=set([object_type]))
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

def combine_object_types(traces_dict, max_iterations):

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
        for i in range(max_iterations):
            missing_object = graph.get_first_missing_object_type()
            if missing_object is None:
                # all activities have been matched up
                break

            possible_choices = traces.get_traces_suitable_for_merging(missing_object)
            next_trace = random.choice(possible_choices)
            next_trace.covered = True
            new_object_id = object_id_generator.get_new_id(next_trace.object_type)
            graph.add_trace(next_trace.object_type, next_trace.sequence, next_trace.id, new_object_id)

        mergedGraphs.append(graph)

    return mergedGraphs, traces


def convert_to_ocel(merged_graph):
    topological_ordering = list(nx.topological_sort(merged_graph.graph))
    rows = []
    for node in topological_ordering:
        row = {"activity": merged_graph.get_node_activity(node)}
        associated_objects = list(merged_graph.get_node_associated_objects(node))
        for object_type in merged_graph.get_node_related_objects(node):
            row[object_type] = [obj for obj in associated_objects if obj.split("_")[0] == object_type]
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    o1 = [["Shared_po", "in", "pa", "Shared_co"], ["Shared_po", "in", "sr", "pa", "Shared_co"]]
    i1 = [["Shared_po", "pi", "Shared_st", "Shared_en", "Shared_co"]]
    r1 = [["Shared_st", "Shared_en"]]

    traces = {"o1": o1, "i1": i1, "r1": r1}

    shared_act_dict = {"Shared_po": {"o1", "i1"}, "Shared_co": {"o1", "i1"}, "Shared_st": {"i1", "r1"},
                       "Shared_en": {"i1", "r1"}}

    graphs, traces = combine_object_types(traces, max_iterations=10)
    print(graphs)

    for graph in graphs:
        print(convert_to_ocel(graph))

    o1 = ["CreateQuotation ApproveQuotation CreateSalesOrder".split(" ")]
    i1 = ["CreateSalesOrder RemoveDeliveryBlock RemoveCreditBlock CreateDelivery CreateCustomerInvoice".split(" "), "CreateSalesOrder CreateDelivery CreateCustomerInvoice".split(" ")]
    r1 = ["CreateSalesOrder ChangeSalesOrderItem CreateDelivery CreateCustomerInvoice".split(" "), "CreateSalesOrder CreateDelivery CreateCustomerInvoice".split(" ")]
    w1 = ["CreateDelivery ExecutePicking InsufficientMaterialFound PostGoodsIssue CreateCustomerInvoice".split(" "), "CreateDelivery ExecutePicking PostGoodsIssue CreateCustomerInvoice".split(" ")]

    traces = {"o1": o1, "i1": i1, "r1": r1, "w1": w1}

    shared_act_dict = {"Shared_po": {"o1", "i1"}, "Shared_co": {"o1", "i1"}, "Shared_st": {"i1", "r1"},
                       "Shared_en": {"i1", "r1"}}

    graphs, traces = combine_object_types(traces, max_iterations=10)

    dataframes = []

    for graph in graphs:
        dataframes.append(convert_to_ocel(graph))

