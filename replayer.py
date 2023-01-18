import pydotplus
from collections import Counter, defaultdict
from enum import Enum
from random import choice, randint
from copy import deepcopy
from graphFlattener import *


class Status(Enum):
    ONGOING = 0
    FINISHED_SUCCESSFULLY = 1
    STOPPED_DUE_TO_CYCLE_LIMIT = 2
    FAILED_DUE_TO_NO_ENABLED_MARKINGS = 3
    FAILED_DUE_TO_MAX_TRACE_LENGTH_EXCEEDED = 4


class Simulation:
    def __init__(self, graph, max_trace_length, max_cycles=2):
        self.graph, self.nodes_with_self_loops = remove_self_loops(graph)
        self.trace_simulations = {}  # id : object
        self.max_cycles = max_cycles
        self.visited_edges = set()
        self.max_trace_length = max_trace_length

        # find start and end nodes
        self.start_node = [node_id for node_id in graph.nodes if graph.nodes.get(node_id)["act_name"] == '"BPMN_START"'][0]
        self.end_node = [node_id for node_id in graph.nodes if graph.nodes.get(node_id)["act_name"] == '"BPMN_END"'][0]

    def start_simulation(self):
        # set marking to start_nodes and then initialize one trace_simulation
        start_marking = defaultdict(lambda: 0)
        for edge in self.graph.out_edges(self.start_node):
            start_marking[edge] = 1

        trace = TraceSimulation(0, self, start_marking, [], set())
        self.trace_simulations[0] = trace

        # by first finishing certain traces, we follow a DFS approach and can check which edges we already successfuly visited
        i = 0
        while i < len(self.trace_simulations):
            while self.trace_simulations[i].status == Status.ONGOING:
                self.trace_simulations[i].play()
            i += 1

    def duplicate_trace(self, id):
        # duplicate trace with identified by id and generate new id for it
        to_be_duplicated = self.trace_simulations[id]
        new_id = max(self.trace_simulations.keys()) + 1
        trace = TraceSimulation(new_id, self, deepcopy(to_be_duplicated.marking), deepcopy(to_be_duplicated.sequence), deepcopy(to_be_duplicated.edge_choices_made))
        self.trace_simulations[new_id] = trace
        return trace

    def get_activity_sequence_representation(self, ignore_self_loops=True):
        traces = []
        for trace in self.trace_simulations.values():
            if trace.status == Status.FINISHED_SUCCESSFULLY:
                if not ignore_self_loops:
                    trace.add_self_loops_to_trace()
                traces.append(trace.map_to_visible_transition_names())
        return traces


class TraceSimulation:
    def __init__(self, id, simulation, marking, sequence=[], edge_choices_made=set()):
        self.id = id
        self.simulation = simulation
        self.marking = marking
        self.graph = self.simulation.graph
        self.sequence = sequence
        self.edge_choices_made = edge_choices_made
        self.status = Status.ONGOING

    def get_enabled_nodes(self):
        edges_with_tokens = set(k for k in self.marking.keys() if self.marking[k] > 0)
        enabled = set()
        for node_id in self.graph.nodes:
            act_name = self.graph.nodes.get(node_id)["act_name"]
            if act_name != '"BPMN_START"':
                # self loops are ignored for computing the required tokens
                required_tokens = set([edge for edge in self.graph.in_edges(node_id) if edge[0] != edge[1]])
                if required_tokens.issubset(edges_with_tokens):
                    enabled.add(node_id)
                elif act_name == '"BPMN_EXCLUSIVE_CHOICE"':
                    # in case of an XOR node, not all ingoing arcs need to be marked
                    if edges_with_tokens.intersection(required_tokens) != set():
                        enabled.add(node_id)
        return enabled

    def get_next_markings_for_xor(self, xor_node_id):
        # each possible next marking for exclusive choice node
        next_markings = []
        base_marking = self.marking

        # choose which incoming edge's token to take
        incoming_edge_with_token = choice([edge for edge in self.graph.in_edges(xor_node_id) if base_marking[edge] > 0])
        base_marking[incoming_edge_with_token] -= 1

        for edge_id in self.graph.out_edges(xor_node_id):
            if edge_id not in self.simulation.visited_edges:
                new_marking = deepcopy(base_marking)
                new_marking[edge_id] += 1
                next_markings.append(new_marking)
                self.edge_choices_made.add(edge_id)
        if len(next_markings) == 0:
            # in case all follow-up choices have been visited already, we just return one, so we don't lose the current trace
            edge_id = choice(list(self.graph.out_edges(xor_node_id)))
            new_marking = deepcopy(base_marking)
            new_marking[edge_id] += 1
            next_markings.append(new_marking)
        return next_markings

    def get_next_marking(self, node_id):
        new_marking = self.marking
        for edge_id in self.graph.in_edges(node_id):
            new_marking[edge_id] -= 1
        for edge_id in self.graph.out_edges(node_id):
            new_marking[edge_id] += 1
        return new_marking

    def play(self):
        if len(self.map_to_visible_transition_names()) > self.simulation.max_trace_length:
            self.status = Status.FAILED_DUE_TO_MAX_TRACE_LENGTH_EXCEEDED
            return

        enabled_nodes = self.get_enabled_nodes()

        # in case all end nodes have been reached, we are done
        if enabled_nodes == set([self.simulation.end_node]):
            self.simulation.visited_edges = self.simulation.visited_edges.union(self.edge_choices_made)
            self.status = Status.FINISHED_SUCCESSFULLY
            return

        # we do not want to pick from end nodes as next activities, so we remove them
        enabled_nodes = enabled_nodes.difference([self.simulation.end_node])
        if enabled_nodes == set():
            self.status = Status.FAILED_DUE_TO_NO_ENABLED_MARKINGS
            return

        # randomly choose next node from enabled non-end nodes
        next_node_id = choice(list(enabled_nodes))
        self.sequence.append(next_node_id)

        # don't consider xor nodes when checking for cycles since they can occur repeatedly without a loop
        nodes_without_xor = [node_id for node_id in self.sequence if self.graph.nodes.get(node_id)["act_name"] != '"BPMN_EXCLUSIVE_CHOICE"']
        counter = Counter(nodes_without_xor).values()
        if len(counter) > 0 and max(counter) > self.simulation.max_cycles:
            self.status = Status.STOPPED_DUE_TO_CYCLE_LIMIT
            return

        if self.graph.nodes.get(next_node_id)["act_name"] == '"BPMN_EXCLUSIVE_CHOICE"':
            # if decision point, duplicate trace, update markings, and play again
            next_markings = self.get_next_markings_for_xor(next_node_id)
            for next_marking in next_markings[:-1]:
                new_trace = self.simulation.duplicate_trace(self.id)
                new_trace.marking = next_marking
#                new_trace.play()
            # last marking is used for original trace
            self.marking = next_markings[-1]
#            self.play()
            return
        else:
            # if node is no decision point, update marking and continue
            self.marking = self.get_next_marking(next_node_id)
#            self.play()
            return

    def map_to_visible_transition_names(self):
        visible_transitions = []
        for node_id in self.sequence:
            activity_name = self.graph.nodes.get(node_id)["act_name"]
            if "BPMN_TASK" in activity_name:
                visible_transitions.append(activity_name)
        return visible_transitions

    def add_self_loops_to_trace(self):
        # for the nodes with self loops insert between 0 and max_cycles occurences of the loop
        trace_nodes_with_loops = set(self.sequence).intersection(self.simulation.nodes_with_self_loops)
        if trace_nodes_with_loops != set():
            new_sequence = []
            iterate_sequence = deepcopy(self.sequence)
            index = 0
            for node in iterate_sequence:
                if node in trace_nodes_with_loops:
                    repeat_times = randint(0, self.simulation.max_cycles)
                    self.sequence = self.sequence[:index] + [node]*repeat_times + self.sequence[index:]
                    index += repeat_times
                index += 1


if __name__ == '__main__':
    graph = """digraph {
    	0[label="BPMN_START"];
    	2[label="BPMN_TASK(CreateDelivery):1338"];
    	3[label="BPMN_TASK(ExecutePicking):1338"];
    	8[label="BPMN_TASK(CreateCustomerInvoice):1338"];
    	4[label="BPMN_EXCLUSIVE_CHOICE"];
    	5[label="BPMN_EXCLUSIVE_CHOICE"];
    	6[label="BPMN_TASK(InsufficientMaterialFound):281"];
    	7[label="BPMN_TASK(PostGoodsIssue):1338"];
    	1[label="BPMN_END"];
    	15[label="BPMN_END"];
    	21[label="BPMN_END"];
    	9[label="BPMN_START"];
    	11[label="BPMN_TASK(CreateQuotation):1338"];
    	12[label="BPMN_TASK(ApproveQuotation):1366"];
    	13[label="BPMN_TASK(CreateSalesOrder):1338"];
    	10[label="BPMN_END"];
    	16[label="BPMN_EXCLUSIVE_CHOICE"];
    	22[label="BPMN_EXCLUSIVE_CHOICE"];
    	14[label="BPMN_START"];
    	17[label="BPMN_EXCLUSIVE_CHOICE"];
    	18[label="BPMN_TASK(RemoveDeliveryBlock):464"];
    	19[label="BPMN_TASK(RemoveCreditBlock):478"];
    	20[label="BPMN_START"];
    	23[label="BPMN_EXCLUSIVE_CHOICE"];
    	24[label="BPMN_TASK(ChangeSalesOrderItem):804"];

    	0 -> 2 [object=0, label=9148];
    	2 -> 3 [object=0, label=9148];
    	2 -> 8 [object=2, label=1142];
    	2 -> 8 [object=3, label=5143];
    	3 -> 4 [object=0, label=9148];
    	4 -> 5 [object=0, label=8012];
    	4 -> 6 [object=0, label=1136];
    	5 -> 7 [object=0, label=9148];
    	6 -> 5 [object=0, label=1136];
    	7 -> 8 [object=0, label=9148];
    	8 -> 1 [object=0, label=9148];
    	8 -> 15 [object=2, label=1142];
    	8 -> 21 [object=3, label=5143];
    	9 -> 11 [object=1, label=1142];
    	11 -> 12 [object=1, label=1142];
    	12 -> 13 [object=1, label=1142];
    	13 -> 10 [object=1, label=1142];
    	13 -> 16 [object=2, label=1142];
    	13 -> 22 [object=3, label=5143];
    	14 -> 13 [object=2, label=1142];
    	16 -> 17 [object=2, label=860];
    	16 -> 18 [object=2, label=282];
    	17 -> 2 [object=2, label=1142];
    	18 -> 19 [object=2, label=282];
    	19 -> 17 [object=2, label=282];
    	20 -> 13 [object=3, label=5143];
    	22 -> 23 [object=3, label=4499];
    	22 -> 24 [object=3, label=644];
    	23 -> 2 [object=3, label=5143];
    	24 -> 23 [object=3, label=644];
    }
    """

    G, object_types = create_graph(graph)
    graphs = {}
    traces = {}
    for ot in object_types:
        graph = flatten_graph(G, ot)
        graphs[ot] = graph

        simulation = Simulation(graph, max_trace_length=20, max_cycles=2)
        simulation.start_simulation()
        traces[ot] = simulation.get_activity_sequence_representation()

    print(traces)
