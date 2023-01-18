import random
from collections import defaultdict
import pandas as pd
import networkx as nx
from replayer import *

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

    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateCustomerInvoice):135138"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_EXCLUSIVE_CHOICE"]; 5[label="BPMN_TASK(SendOverdueNotice):78174"]; 6[label="BPMN_TASK(ClearCustomerInvoice):135138"]; 1[label="BPMN_END"]; 7[label="BPMN_START"]; 9[label="BPMN_TASK(CreateDelivery):135138"]; 10[label="BPMN_TASK(ExecutePicking):135138"]; 11[label="BPMN_TASK(InsufficientMaterialFound):28381"]; 12[label="BPMN_TASK(PostGoodsIssue):135138"]; 8[label="BPMN_END"]; 13[label="BPMN_START"]; 15[label="BPMN_TASK(PostGoodsReceipt):135138"]; 16[label="BPMN_TASK(ReceiveVendorInvoice):135138"]; 14[label="BPMN_END"]; 17[label="BPMN_START"]; 19[label="BPMN_TASK(CreatePurchaseOrder):135138"]; 20[label="BPMN_TASK(ApprovePurchaseOrder):130896"]; 21[label="BPMN_TASK(SendPurchaseOrder):130896"]; 22[label="BPMN_TASK(SendDeliveryOverdueNotice):28381"]; 18[label="BPMN_END"]; 23[label="BPMN_START"]; 25[label="BPMN_TASK(CreateQuotation):135138"]; 26[label="BPMN_TASK(ApproveQuotation):137966"]; 27[label="BPMN_TASK(CreateSalesOrder):135138"]; 24[label="BPMN_END"]; 28[label="BPMN_START"]; 30[label="BPMN_EXCLUSIVE_CHOICE"]; 31[label="BPMN_EXCLUSIVE_CHOICE"]; 32[label="BPMN_TASK(RemoveCreditBlock):48278"]; 33[label="BPMN_TASK(RemoveDeliveryBlock):46864"]; 29[label="BPMN_END"]; 34[label="BPMN_START"]; 35[label="BPMN_END"]; 36[label="BPMN_START"]; 37[label="BPMN_END"]; 38[label="BPMN_START"]; 40[label="BPMN_EXCLUSIVE_CHOICE"]; 42[label="BPMN_TASK(ReceiveOverdueNotice):44036"]; 41[label="BPMN_EXCLUSIVE_CHOICE"]; 43[label="BPMN_TASK(SetPaymentBlock):42723"]; 44[label="BPMN_TASK(RemovePaymentBlock):42723"]; 45[label="BPMN_TASK(ClearVendorInvoice):135138"]; 39[label="BPMN_END"]; 0 -> 2 [object=0, label=72720]; 2 -> 3 [object=0, label=72720]; 3 -> 4 [object=0, label=36360]; 3 -> 5 [object=0, label=36360]; 5 -> 4 [object=0, label=36360]; 4 -> 6 [object=0, label=72720]; 6 -> 1 [object=0, label=72720]; 7 -> 9 [object=1, label=117059]; 9 -> 10 [object=1, label=117059]; 10 -> 11 [object=1, label=117059]; 11 -> 12 [object=1, label=117059]; 12 -> 2 [object=1, label=117059]; 2 -> 8 [object=1, label=117059]; 13 -> 15 [object=2, label=14544]; 15 -> 10 [object=2, label=14544]; 10 -> 11 [object=2, label=14544]; 11 -> 12 [object=2, label=14544]; 12 -> 16 [object=2, label=14544]; 16 -> 14 [object=2, label=14544]; 17 -> 19 [object=3, label=14544]; 19 -> 20 [object=3, label=14544]; 20 -> 21 [object=3, label=14544]; 21 -> 22 [object=3, label=14544]; 22 -> 15 [object=3, label=14544]; 15 -> 16 [object=3, label=14544]; 16 -> 18 [object=3, label=14544]; 23 -> 25 [object=4, label=14544]; 25 -> 26 [object=4, label=14544]; 26 -> 27 [object=4, label=14544]; 27 -> 24 [object=4, label=14544]; 28 -> 27 [object=5, label=14544]; 27 -> 30 [object=5, label=14544]; 30 -> 31 [object=5, label=7272]; 30 -> 32 [object=5, label=7272]; 32 -> 33 [object=5, label=7272]; 33 -> 31 [object=5, label=7272]; 31 -> 9 [object=5, label=14544]; 9 -> 2 [object=5, label=14544]; 2 -> 29 [object=5, label=14544]; 34 -> 27 [object=6, label=14544]; 27 -> 9 [object=6, label=14544]; 9 -> 2 [object=6, label=14544]; 2 -> 35 [object=6, label=14544]; 36 -> 19 [object=7, label=4040]; 19 -> 37 [object=7, label=4040]; 38 -> 16 [object=8, label=72922]; 16 -> 40 [object=8, label=72922]; 40 -> 42 [object=8, label=36057]; 42 -> 41 [object=8, label=36057]; 40 -> 43 [object=8, label=36865]; 43 -> 44 [object=8, label=36865]; 44 -> 41 [object=8, label=36865]; 41 -> 45 [object=8, label=72922]; 45 -> 39 [object=8, label=72922];}"""
    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateCustomerInvoice):135138"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_TASK(SendOverdueNotice):78174"]; 5[label="BPMN_TASK(ClearCustomerInvoice):135138"]; 1[label="BPMN_END"]; 6[label="BPMN_START"]; 8[label="BPMN_TASK(CreateDelivery):135138"]; 9[label="BPMN_TASK(ExecutePicking):135138"]; 10[label="BPMN_EXCLUSIVE_CHOICE"]; 11[label="BPMN_EXCLUSIVE_CHOICE"]; 12[label="BPMN_TASK(InsufficientMaterialFound):28381"]; 13[label="BPMN_TASK(PostGoodsIssue):135138"]; 7[label="BPMN_END"]; 14[label="BPMN_START"]; 16[label="BPMN_TASK(CreateQuotation):135138"]; 17[label="BPMN_TASK(ApproveQuotation):137966"]; 18[label="BPMN_TASK(CreateSalesOrder):135138"]; 15[label="BPMN_END"]; 19[label="BPMN_START"]; 21[label="BPMN_EXCLUSIVE_CHOICE"]; 22[label="BPMN_EXCLUSIVE_CHOICE"]; 23[label="BPMN_PARALLEL"]; 25[label="BPMN_EXCLUSIVE_CHOICE"]; 26[label="BPMN_EXCLUSIVE_CHOICE"]; 27[label="BPMN_TASK(ChangeShipTo):4242"]; 28[label="BPMN_TASK(RemoveDeliveryBlock):46864"]; 24[label="BPMN_PARALLEL"]; 29[label="BPMN_TASK(RemoveCreditBlock):48278"]; 30[label="BPMN_TASK(ChangeSoldTo):1414"]; 20[label="BPMN_END"]; 31[label="BPMN_START"]; 33[label="BPMN_EXCLUSIVE_CHOICE"]; 34[label="BPMN_EXCLUSIVE_CHOICE"]; 35[label="BPMN_TASK(ChangeSalesOrderItem):81305"]; 32[label="BPMN_END"]; 0 -> 2 [object=0, label=135138]; 2 -> 3 [object=0, label=135138]; 3 -> 4 [object=0, label=78174]; 4 -> 3 [object=0, label=78174]; 3 -> 5 [object=0, label=135138]; 5 -> 1 [object=0, label=135138]; 6 -> 8 [object=1, label=1086053]; 8 -> 9 [object=1, label=1086053]; 9 -> 10 [object=1, label=1086053]; 10 -> 11 [object=1, label=856884]; 10 -> 12 [object=1, label=229169]; 12 -> 11 [object=1, label=229169]; 11 -> 13 [object=1, label=1086053]; 13 -> 2 [object=1, label=1086053]; 2 -> 7 [object=1, label=1086053]; 14 -> 16 [object=2, label=135138]; 16 -> 17 [object=2, label=135138]; 17 -> 17 [object=2, label=2828]; 17 -> 18 [object=2, label=135138]; 18 -> 15 [object=2, label=135138]; 19 -> 18 [object=3, label=135138]; 18 -> 21 [object=3, label=135138]; 21 -> 22 [object=3, label=86860]; 21 -> 23 [object=3, label=46864]; 23 -> 25 [object=3, label=46864]; 25 -> 26 [object=3, label=42622]; 25 -> 27 [object=3, label=4242]; 27 -> 26 [object=3, label=4242]; 26 -> 28 [object=3, label=46864]; 28 -> 24 [object=3, label=46864]; 23 -> 29 [object=3, label=46864]; 29 -> 29 [object=3, label=1414]; 29 -> 24 [object=3, label=46864]; 24 -> 22 [object=3, label=46864]; 21 -> 30 [object=3, label=1414]; 30 -> 22 [object=3, label=1414]; 22 -> 8 [object=3, label=135138]; 8 -> 2 [object=3, label=135138]; 2 -> 20 [object=3, label=135138]; 31 -> 18 [object=4, label=610848]; 18 -> 33 [object=4, label=610848]; 33 -> 34 [object=4, label=532876]; 33 -> 35 [object=4, label=77972]; 35 -> 34 [object=4, label=77972]; 34 -> 8 [object=4, label=610848]; 8 -> 2 [object=4, label=610848]; 2 -> 32 [object=4, label=610848];}"""

    G, object_types = create_graph(graph)
    graphs = {}
    traces = {}
    for ot in object_types:
        graph = flatten_graph(G, ot)
        graphs[ot] = graph

        simulation = Simulation(graph, max_trace_length=20, max_cycles=2)
        simulation.start_simulation()
        traces[ot] = simulation.get_activity_sequence_representation(ignore_self_loops=False)

    graphs, traces = combine_object_types(traces, max_iterations=50)

    dataframes = []

    for graph in graphs:
        dataframes.append(convert_to_ocel(graph))

    print(dataframes)
