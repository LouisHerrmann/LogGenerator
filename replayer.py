from collections import Counter, defaultdict
from enum import Enum
from random import choice, randint
from graphFlattener import *


class Status(Enum):
    ONGOING = 0
    FINISHED_SUCCESSFULLY = 1
    STOPPED_DUE_TO_CYCLE_LIMIT = 2
    FAILED_DUE_TO_NO_ENABLED_MARKINGS = 3
    FAILED_DUE_TO_MAX_TRACE_LENGTH_EXCEEDED = 4


class Simulation:
    def __init__(self, graph, max_trace_length, max_cycles=None):
        if max_cycles is None:
            max_cycles=2
        self.graph, self.nodes_with_self_loops = remove_self_loops(graph)
        self.trace_simulations = {}  # id : object
        self.max_cycles = max_cycles
        self.max_trace_length = max_trace_length

        # find start and end nodes
        self.start_node = [node_id for node_id in graph.nodes
                           if graph.nodes.get(node_id)["act_name"] == '"BPMN_START"'][0]
        self.end_node = [node_id for node_id in graph.nodes
                         if graph.nodes.get(node_id)["act_name"] == '"BPMN_END"'][0]

    def start_simulation(self):
        # set marking to start_nodes and then initialize one trace_simulation
        start_marking = defaultdict(lambda: 0)
        for edge in self.graph.out_edges(self.start_node):
            start_marking[edge] = 1

        trace = TraceSimulation(0, self, start_marking, [], set())
        self.trace_simulations[0] = trace

        # by first finishing certain traces, we follow a DFS approach and can check
        # which edges we already successfully visited (currently are not checking this since lead to problems in past)
        # alternative: bfs approach can be achieved by immediately playing all traces after duplication
        # (can likely be parallelized better)
        i = 0
        while i < len(self.trace_simulations):
            while self.trace_simulations[i].status == Status.ONGOING:
                self.trace_simulations[i].play()
            i += 1

    def duplicate_trace(self, id):
        # duplicate trace with identified by id and generate new id for it
        to_be_duplicated = self.trace_simulations[id]
        new_id = max(self.trace_simulations.keys()) + 1
        trace = TraceSimulation(new_id, self, deepcopy(to_be_duplicated.marking),
                                deepcopy(to_be_duplicated.sequence),
                                deepcopy(to_be_duplicated.edge_choices_made))
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

    def check_node_coverage(self):
        counter = {node: 0 for node in self.graph.nodes if node != self.start_node and node != self.end_node}
        for trace in self.trace_simulations.values():
            for node in trace.sequence:
                counter[node] += 1

        return counter

class TraceSimulation:
    def __init__(self, id, simulation, marking, sequence=None, edge_choices_made=None):
        if sequence is None:
            sequence = []
        if edge_choices_made is None:
            edge_choices_made = set()
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
        incoming_edge_with_token = choice([edge for edge in self.graph.in_edges(xor_node_id)
                                           if base_marking[edge] > 0])
        base_marking[incoming_edge_with_token] -= 1

        for edge_id in self.graph.out_edges(xor_node_id):
            new_marking = deepcopy(base_marking)
            new_marking[edge_id] += 1
            next_markings.append(new_marking)
            self.edge_choices_made.add(edge_id)
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
        if enabled_nodes == {self.simulation.end_node}:
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
        nodes_without_xor = [node_id for node_id in self.sequence
                             if self.graph.nodes.get(node_id)["act_name"] != '"BPMN_EXCLUSIVE_CHOICE"']
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

    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateCustomerInvoice):135138"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_TASK(SendOverdueNotice):78174"]; 5[label="BPMN_TASK(ClearCustomerInvoice):135138"]; 1[label="BPMN_END"]; 6[label="BPMN_START"]; 8[label="BPMN_TASK(CreateDelivery):135138"]; 9[label="BPMN_TASK(ExecutePicking):135138"]; 10[label="BPMN_EXCLUSIVE_CHOICE"]; 11[label="BPMN_EXCLUSIVE_CHOICE"]; 12[label="BPMN_TASK(InsufficientMaterialFound):28381"]; 13[label="BPMN_TASK(PostGoodsIssue):135138"]; 7[label="BPMN_END"]; 14[label="BPMN_START"]; 16[label="BPMN_TASK(PostGoodsReceipt):135138"]; 17[label="BPMN_EXCLUSIVE_CHOICE"]; 18[label="BPMN_EXCLUSIVE_CHOICE"]; 19[label="BPMN_TASK(ReceiveVendorInvoice):135138"]; 15[label="BPMN_END"]; 20[label="BPMN_START"]; 21[label="BPMN_END"]; 22[label="BPMN_START"]; 24[label="BPMN_TASK(CreatePurchaseOrder):135138"]; 25[label="BPMN_EXCLUSIVE_CHOICE"]; 27[label="BPMN_EXCLUSIVE_CHOICE"]; 26[label="BPMN_TASK(SendPurchaseOrder):130896"]; 28[label="BPMN_EXCLUSIVE_CHOICE"]; 29[label="BPMN_TASK(ApprovePurchaseOrder):130896"]; 30[label="BPMN_EXCLUSIVE_CHOICE"]; 31[label="BPMN_TASK(SendDeliveryOverdueNotice):28381"]; 23[label="BPMN_END"]; 32[label="BPMN_START"]; 34[label="BPMN_TASK(CreateQuotation):135138"]; 35[label="BPMN_TASK(ApproveQuotation):137966"]; 36[label="BPMN_TASK(CreateSalesOrder):135138"]; 33[label="BPMN_END"]; 37[label="BPMN_START"]; 39[label="BPMN_EXCLUSIVE_CHOICE"]; 40[label="BPMN_EXCLUSIVE_CHOICE"]; 41[label="BPMN_PARALLEL"]; 43[label="BPMN_EXCLUSIVE_CHOICE"]; 44[label="BPMN_EXCLUSIVE_CHOICE"]; 45[label="BPMN_TASK(ChangeShipTo):4242"]; 46[label="BPMN_TASK(RemoveDeliveryBlock):46864"]; 42[label="BPMN_PARALLEL"]; 47[label="BPMN_TASK(RemoveCreditBlock):48278"]; 48[label="BPMN_TASK(ChangeSoldTo):1414"]; 38[label="BPMN_END"]; 49[label="BPMN_START"]; 51[label="BPMN_EXCLUSIVE_CHOICE"]; 52[label="BPMN_EXCLUSIVE_CHOICE"]; 53[label="BPMN_TASK(ChangeSalesOrderItem):81305"]; 50[label="BPMN_END"]; 54[label="BPMN_START"]; 55[label="BPMN_END"]; 56[label="BPMN_START"]; 58[label="BPMN_EXCLUSIVE_CHOICE"]; 60[label="BPMN_EXCLUSIVE_CHOICE"]; 61[label="BPMN_TASK(ChangePaymentTerms):12726"]; 62[label="BPMN_EXCLUSIVE_CHOICE"]; 63[label="BPMN_TASK(ChangeCashDiscountDueDate):7070"]; 59[label="BPMN_EXCLUSIVE_CHOICE"]; 64[label="BPMN_TASK(ReceiveOverdueNotice):44036"]; 65[label="BPMN_TASK(SetPaymentBlock):42723"]; 66[label="BPMN_TASK(RemovePaymentBlock):42723"]; 67[label="BPMN_TASK(ClearVendorInvoice):135138"]; 57[label="BPMN_END"]; 68[label="BPMN_START"]; 70[label="BPMN_EXCLUSIVE_CHOICE"]; 72[label="BPMN_EXCLUSIVE_CHOICE"]; 73[label="BPMN_EXCLUSIVE_CHOICE"]; 71[label="BPMN_EXCLUSIVE_CHOICE"]; 69[label="BPMN_END"]; 0 -> 2 [object=0, label=135138]; 2 -> 3 [object=0, label=135138]; 3 -> 4 [object=0, label=78174]; 4 -> 3 [object=0, label=78174]; 3 -> 5 [object=0, label=135138]; 5 -> 1 [object=0, label=135138]; 6 -> 8 [object=1, label=135138]; 8 -> 9 [object=1, label=135138]; 9 -> 10 [object=1, label=135138]; 10 -> 11 [object=1, label=106757]; 10 -> 12 [object=1, label=28381]; 12 -> 11 [object=1, label=28381]; 11 -> 13 [object=1, label=135138]; 13 -> 2 [object=1, label=135138]; 2 -> 7 [object=1, label=135138]; 14 -> 16 [object=2, label=605697]; 16 -> 9 [object=2, label=605697]; 9 -> 17 [object=2, label=605697]; 17 -> 18 [object=2, label=479346]; 17 -> 12 [object=2, label=126351]; 12 -> 18 [object=2, label=126351]; 18 -> 13 [object=2, label=605697]; 13 -> 19 [object=2, label=605697]; 19 -> 15 [object=2, label=605697]; 20 -> 13 [object=3, label=880215]; 13 -> 21 [object=3, label=880215]; 22 -> 24 [object=4, label=4242]; 24 -> 25 [object=4, label=4242]; 25 -> 27 [object=4, label=4242]; 25 -> 26 [object=4, label=0]; 26 -> 27 [object=4, label=0]; 27 -> 28 [object=4, label=4242]; 27 -> 29 [object=4, label=0]; 29 -> 28 [object=4, label=0]; 28 -> 30 [object=4, label=4242]; 28 -> 31 [object=4, label=0]; 31 -> 30 [object=4, label=0]; 30 -> 16 [object=4, label=4242]; 16 -> 19 [object=4, label=4242]; 19 -> 23 [object=4, label=4242]; 32 -> 34 [object=5, label=135138]; 34 -> 35 [object=5, label=135138]; 35 -> 35 [object=5, label=2828]; 35 -> 36 [object=5, label=135138]; 36 -> 33 [object=5, label=135138]; 37 -> 36 [object=6, label=135138]; 36 -> 39 [object=6, label=135138]; 39 -> 40 [object=6, label=86860]; 39 -> 41 [object=6, label=46864]; 41 -> 43 [object=6, label=46864]; 43 -> 44 [object=6, label=42622]; 43 -> 45 [object=6, label=4242]; 45 -> 44 [object=6, label=4242]; 44 -> 46 [object=6, label=46864]; 46 -> 42 [object=6, label=46864]; 41 -> 47 [object=6, label=46864]; 47 -> 47 [object=6, label=1414]; 47 -> 42 [object=6, label=46864]; 42 -> 40 [object=6, label=46864]; 39 -> 48 [object=6, label=1414]; 48 -> 40 [object=6, label=1414]; 40 -> 8 [object=6, label=135138]; 8 -> 2 [object=6, label=135138]; 2 -> 38 [object=6, label=135138]; 49 -> 36 [object=7, label=610848]; 36 -> 51 [object=7, label=610848]; 51 -> 52 [object=7, label=532876]; 51 -> 53 [object=7, label=77972]; 53 -> 52 [object=7, label=77972]; 52 -> 8 [object=7, label=610848]; 8 -> 2 [object=7, label=610848]; 2 -> 50 [object=7, label=610848]; 54 -> 24 [object=8, label=5050]; 24 -> 55 [object=8, label=5050]; 56 -> 19 [object=9, label=135138]; 19 -> 58 [object=9, label=135138]; 58 -> 60 [object=9, label=92415]; 60 -> 61 [object=9, label=12726]; 61 -> 62 [object=9, label=12726]; 62 -> 60 [object=9, label=5656]; 62 -> 63 [object=9, label=7070]; 63 -> 60 [object=9, label=7070]; 60 -> 59 [object=9, label=48379]; 60 -> 64 [object=9, label=44036]; 64 -> 59 [object=9, label=44036]; 58 -> 65 [object=9, label=42723]; 65 -> 66 [object=9, label=42723]; 66 -> 59 [object=9, label=42723]; 59 -> 67 [object=9, label=135138]; 67 -> 57 [object=9, label=135138]; 68 -> 19 [object=10, label=670741]; 19 -> 70 [object=10, label=670741]; 70 -> 72 [object=10, label=459247]; 72 -> 61 [object=10, label=62721]; 61 -> 73 [object=10, label=62721]; 73 -> 72 [object=10, label=27977]; 73 -> 63 [object=10, label=34744]; 63 -> 72 [object=10, label=34744]; 72 -> 71 [object=10, label=242501]; 72 -> 64 [object=10, label=216746]; 64 -> 71 [object=10, label=216746]; 70 -> 65 [object=10, label=211494]; 65 -> 66 [object=10, label=211494]; 66 -> 71 [object=10, label=211494]; 71 -> 67 [object=10, label=670741]; 67 -> 69 [object=10, label=670741];}"""

    G, object_types = create_graph(graph)
    graphs = {}
    traces = {}
    for ot in object_types:
        graph = flatten_graph(G, ot)
        graphs[ot] = graph

        simulation = Simulation(graph, max_trace_length=20, max_cycles=2)
        simulation.start_simulation()
        traces[ot] = simulation.get_activity_sequence_representation()
        print(simulation.check_node_coverage())
