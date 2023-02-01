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
    def __init__(self, graph, max_trace_length, max_cycles=None, minimize_for_node_coverage=None):
        if max_cycles is None:
            max_cycles = 2
        if minimize_for_node_coverage is None:
            minimize_for_node_coverage = False
        self.graph, self.nodes_with_self_loops = remove_self_loops(graph)
        self.trace_simulations = {}  # id : object
        self.max_cycles = max_cycles
        self.max_trace_length = max_trace_length
        self.minimize_for_node_coverage = minimize_for_node_coverage

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
            if self.minimize_for_node_coverage and self.get_uncovered_nodes_coverage() == set():
                # in case all nodes have been covered already, and we don't care about paths, we stop
                break
            while self.trace_simulations[i].status == Status.ONGOING:
                self.trace_simulations[i].play()
            i += 1

    def duplicate_trace(self, id):
        # duplicate trace identified by id and generate new id
        to_be_duplicated = self.trace_simulations[id]
        new_id = max(self.trace_simulations.keys()) + 1
        trace = TraceSimulation(new_id, self, deepcopy(to_be_duplicated.marking),
                                deepcopy(to_be_duplicated.sequence),
                                deepcopy(to_be_duplicated.edge_choices_made))
        self.trace_simulations[new_id] = trace
        return trace

    def get_activity_sequence_representation(self, ignore_self_loops=True):
        # convert all traces from node sequences to sequence of visible transition names + optionally add self loops
        traces = []
        for trace in self.trace_simulations.values():
            if trace.status == Status.FINISHED_SUCCESSFULLY:
                if not ignore_self_loops:
                    trace.add_self_loops_to_trace()
                traces.append(trace.map_to_visible_transition_names())
        return traces

    def get_uncovered_nodes_coverage(self):
        # return nodes that have not yet been visited by any completed traces
        all_nodes = set(self.graph.nodes).difference({self.start_node, self.end_node})
        covered_nodes = set()
        for trace_sequence in [t.sequence for t in self.trace_simulations.values()
                               if t.status == Status.FINISHED_SUCCESSFULLY]:
            covered_nodes = covered_nodes.union(trace_sequence)
        return all_nodes.difference(covered_nodes)


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

        # in case we would only want to check cycles with activity names, we would have to disregard XOR nodes
        # e.g. consider XOR to activity and back to XOR (XOR cycle but no activity cycle). However, we still would
        # need to check for XOR loops to avoid infinite XOR loops
        #        nodes_without_xor = [node_id for node_id in self.sequence
        #                             if self.graph.nodes.get(node_id)["act_name"] != '"BPMN_EXCLUSIVE_CHOICE"']
        counter = Counter(self.sequence).values()
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
            #            self.play()    # immediately playing duplicated traces corresponds to a BFS-based approach
            return
        else:
            # if node is no decision point, update marking and continue
            self.marking = self.get_next_marking(next_node_id)
            #            self.play()    # immediately playing duplicated traces corresponds to a BFS-based approach
            return

    def map_to_visible_transition_names(self):
        visible_transitions = []
        for node_id in self.sequence:
            activity_name = self.graph.nodes.get(node_id)["act_name"]
            if "BPMN_TASK" in activity_name:
                visible_transitions.append(activity_name)
        return visible_transitions

    def add_self_loops_to_trace(self):
        # for the nodes with self loops insert between 0 and max_cycles occurrences of the loop
        trace_nodes_with_loops = set(self.sequence).intersection(self.simulation.nodes_with_self_loops)
        if trace_nodes_with_loops != set():
            new_sequence = []
            iterate_sequence = deepcopy(self.sequence)
            index = 0
            for node in iterate_sequence:
                if node in trace_nodes_with_loops:
                    repeat_times = randint(0, self.simulation.max_cycles)
                    self.sequence = self.sequence[:index] + [node] * repeat_times + self.sequence[index:]
                    index += repeat_times
                index += 1


if __name__ == '__main__':

    # trickier graphs
#    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreatePurchaseOrder):58155"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_EXCLUSIVE_CHOICE"]; 5[label="BPMN_TASK(AddPurchaseOrderItem):18079"]; 6[label="BPMN_TASK(CreateContract):47"]; 7[label="BPMN_EXCLUSIVE_CHOICE"]; 8[label="BPMN_EXCLUSIVE_CHOICE"]; 9[label="BPMN_TASK(ChangeContract):4"]; 10[label="BPMN_TASK(ChangeContractItem):6"]; 11[label="BPMN_TASK(CreatePurchaseRequisitionAutomatically):268058"]; 1[label="BPMN_END"]; 15[label="BPMN_START"]; 17[label="BPMN_TASK(CreateInvoiceReceipt):194541"]; 18[label="BPMN_TASK(PostVendorInvoice):28239"]; 19[label="BPMN_EXCLUSIVE_CHOICE"]; 20[label="BPMN_EXCLUSIVE_CHOICE"]; 21[label="BPMN_TASK(ReverseInvoiceReceipt):24"]; 22[label="BPMN_TASK(CreateCreditMemoItem):1603"]; 16[label="BPMN_END"]; 23[label="BPMN_START"]; 25[label="BPMN_EXCLUSIVE_CHOICE"]; 26[label="BPMN_EXCLUSIVE_CHOICE"]; 27[label="BPMN_EXCLUSIVE_CHOICE"]; 29[label="BPMN_EXCLUSIVE_CHOICE"]; 30[label="BPMN_TASK(ReactivatePurchaseOrderItem):826"]; 32[label="BPMN_EXCLUSIVE_CHOICE"]; 31[label="BPMN_TASK(BlockPurchaseOrderItem):222"]; 28[label="BPMN_EXCLUSIVE_CHOICE"]; 33[label="BPMN_TASK(RestorePurchaseOrderItem):656"]; 34[label="BPMN_TASK(ChangePrice):2093"]; 35[label="BPMN_TASK(CreateScheduleLineAgreement):1593511"]; 36[label="BPMN_TASK(ReceiveAdvancedShipmentNotice):240879"]; 37[label="BPMN_TASK(ChangePurchaseOrderItem):3917"]; 38[label="BPMN_TASK(DeletePurchaseOrderItem):14439"]; 39[label="BPMN_TASK(PostGoodsReceipt):393176"]; 40[label="BPMN_TASK(ReceiveOrderConfirmation):183515"]; 41[label="BPMN_TASK(ReverseGoodsReceipt):355"]; 24[label="BPMN_END"]; 44[label="BPMN_START"]; 45[label="BPMN_END"]; 48[label="BPMN_START"]; 50[label="BPMN_EXCLUSIVE_CHOICE"]; 52[label="BPMN_EXCLUSIVE_CHOICE"]; 51[label="BPMN_EXCLUSIVE_CHOICE"]; 53[label="BPMN_TASK(ChangeScheduleLineAgreement):222334"]; 54[label="BPMN_TASK(MassConvertPurchaseRequisitionToPurchaseOrder):3387"]; 49[label="BPMN_END"]; 0 -> 2 [object=0, label=11]; 2 -> 3 [object=0, label=11]; 3 -> 4 [object=0, label=9]; 3 -> 5 [object=0, label=2]; 5 -> 4 [object=0, label=2]; 4 -> 6 [object=0, label=11]; 6 -> 7 [object=0, label=11]; 7 -> 8 [object=0, label=8]; 7 -> 9 [object=0, label=1]; 9 -> 9 [object=0, label=1]; 9 -> 8 [object=0, label=1]; 7 -> 10 [object=0, label=1]; 10 -> 10 [object=0, label=5]; 10 -> 8 [object=0, label=1]; 7 -> 11 [object=0, label=1]; 11 -> 11 [object=0, label=2]; 11 -> 8 [object=0, label=1]; 8 -> 1 [object=0, label=11]; 15 -> 17 [object=2, label=193731]; 17 -> 18 [object=2, label=193731]; 18 -> 19 [object=2, label=193731]; 19 -> 20 [object=2, label=193714]; 19 -> 21 [object=2, label=17]; 21 -> 22 [object=2, label=17]; 22 -> 20 [object=2, label=17]; 20 -> 16 [object=2, label=193731]; 23 -> 25 [object=3, label=185452]; 25 -> 26 [object=3, label=49196]; 25 -> 2 [object=3, label=136256]; 2 -> 26 [object=3, label=136256]; 26 -> 27 [object=3, label=174887]; 26 -> 5 [object=3, label=10565]; 5 -> 27 [object=3, label=10565]; 27 -> 29 [object=3, label=1075648]; 27 -> 30 [object=3, label=298]; 30 -> 29 [object=3, label=298]; 29 -> 32 [object=3, label=237]; 29 -> 31 [object=3, label=51]; 31 -> 32 [object=3, label=51]; 32 -> 28 [object=3, label=53]; 32 -> 33 [object=3, label=235]; 33 -> 28 [object=3, label=235]; 29 -> 34 [object=3, label=1130]; 34 -> 28 [object=3, label=1130]; 29 -> 17 [object=3, label=193735]; 17 -> 28 [object=3, label=193735]; 29 -> 35 [object=3, label=756494]; 35 -> 28 [object=3, label=756494]; 29 -> 36 [object=3, label=124299]; 36 -> 28 [object=3, label=124299]; 27 -> 37 [object=3, label=2001]; 37 -> 28 [object=3, label=2001]; 27 -> 38 [object=3, label=325]; 38 -> 28 [object=3, label=325]; 27 -> 39 [object=3, label=192791]; 39 -> 28 [object=3, label=192791]; 27 -> 40 [object=3, label=104786]; 40 -> 28 [object=3, label=104786]; 28 -> 27 [object=3, label=1190333]; 28 -> 41 [object=3, label=64]; 41 -> 41 [object=3, label=7]; 41 -> 27 [object=3, label=64]; 28 -> 24 [object=3, label=185452]; 44 -> 21 [object=5, label=17]; 21 -> 45 [object=5, label=17]; 48 -> 50 [object=7, label=179798]; 50 -> 2 [object=7, label=158079]; 2 -> 52 [object=7, label=158079]; 52 -> 51 [object=7, label=111949]; 52 -> 5 [object=7, label=7793]; 5 -> 35 [object=7, label=7793]; 35 -> 51 [object=7, label=7793]; 52 -> 53 [object=7, label=38337]; 53 -> 51 [object=7, label=38337]; 50 -> 54 [object=7, label=21719]; 54 -> 54 [object=7, label=21719]; 54 -> 51 [object=7, label=21719]; 51 -> 49 [object=7, label=179798];}"""
#    graph = """digraph { 2[label="BPMN_START"]; 4[label="BPMN_PARALLEL"]; 6[label="BPMN_EXCLUSIVE_CHOICE"]; 7[label="BPMN_TASK(AddDeliveryItems):543735"]; 5[label="BPMN_PARALLEL"]; 8[label="BPMN_TASK(CreateDelivery):860495"]; 9[label="BPMN_EXCLUSIVE_CHOICE"]; 10[label="BPMN_TASK(CreateCustomerInvoice):688953"]; 3[label="BPMN_END"]; 11[label="BPMN_START"]; 13[label="BPMN_PARALLEL"]; 14[label="BPMN_PARALLEL"]; 15[label="BPMN_EXCLUSIVE_CHOICE"]; 12[label="BPMN_END"]; 20[label="BPMN_START"]; 22[label="BPMN_TASK(CreateSalesOrder):698719"]; 23[label="BPMN_EXCLUSIVE_CHOICE"]; 24[label="BPMN_TASK(SetDeliveryBlock):24002"]; 25[label="BPMN_PARALLEL"]; 27[label="BPMN_TASK(ChangeSalesOrder):154735"]; 26[label="BPMN_PARALLEL"]; 28[label="BPMN_EXCLUSIVE_CHOICE"]; 29[label="BPMN_TASK(RemoveDeliveryBlock):766880"]; 30[label="BPMN_TASK(ReleaseCreditHold):143048"]; 31[label="BPMN_EXCLUSIVE_CHOICE"]; 32[label="BPMN_TASK(SetCreditHold):13564"]; 33[label="BPMN_EXCLUSIVE_CHOICE"]; 34[label="BPMN_TASK(PassCredit):15068"]; 35[label="BPMN_TASK(AddSalesOrderItems):16742"]; 21[label="BPMN_END"]; 36[label="BPMN_START"]; 38[label="BPMN_EXCLUSIVE_CHOICE"]; 39[label="BPMN_EXCLUSIVE_CHOICE"]; 40[label="BPMN_EXCLUSIVE_CHOICE"]; 41[label="BPMN_TASK(SetRejectionReason):228098"]; 42[label="BPMN_TASK(CancelRejectionReason):24280"]; 43[label="BPMN_PARALLEL"]; 44[label="BPMN_PARALLEL"]; 45[label="BPMN_EXCLUSIVE_CHOICE"]; 46[label="BPMN_TASK(ChangeSalesOrderScheduleLine):2064813"]; 47[label="BPMN_TASK(ChangeSalesOrderItem):190696"]; 37[label="BPMN_END"]; 48[label="BPMN_START"]; 49[label="BPMN_END"]; 2 -> 4 [object=1, label=42]; 4 -> 6 [object=1, label=42]; 6 -> 7 [object=1, label=52]; 7 -> 6 [object=1, label=52]; 6 -> 5 [object=1, label=42]; 4 -> 8 [object=1, label=42]; 8 -> 9 [object=1, label=42]; 9 -> 10 [object=1, label=41]; 10 -> 9 [object=1, label=41]; 9 -> 5 [object=1, label=42]; 5 -> 3 [object=1, label=42]; 11 -> 13 [object=2, label=71]; 13 -> 8 [object=2, label=71]; 8 -> 14 [object=2, label=71]; 13 -> 7 [object=2, label=71]; 7 -> 14 [object=2, label=71]; 14 -> 15 [object=2, label=71]; 15 -> 10 [object=2, label=34]; 10 -> 15 [object=2, label=34]; 15 -> 12 [object=2, label=71]; 20 -> 22 [object=5, label=25]; 22 -> 23 [object=5, label=25]; 23 -> 24 [object=5, label=2]; 24 -> 23 [object=5, label=2]; 23 -> 25 [object=5, label=25]; 25 -> 27 [object=5, label=25]; 27 -> 27 [object=5, label=30]; 27 -> 26 [object=5, label=25]; 25 -> 28 [object=5, label=25]; 28 -> 29 [object=5, label=38]; 29 -> 28 [object=5, label=38]; 28 -> 30 [object=5, label=18]; 30 -> 28 [object=5, label=18]; 28 -> 31 [object=5, label=25]; 31 -> 32 [object=5, label=6]; 32 -> 31 [object=5, label=6]; 31 -> 26 [object=5, label=25]; 25 -> 8 [object=5, label=25]; 8 -> 8 [object=5, label=18]; 8 -> 26 [object=5, label=25]; 25 -> 33 [object=5, label=25]; 33 -> 34 [object=5, label=21]; 34 -> 33 [object=5, label=21]; 33 -> 35 [object=5, label=9]; 35 -> 33 [object=5, label=9]; 33 -> 26 [object=5, label=25]; 26 -> 21 [object=5, label=25]; 36 -> 38 [object=6, label=60]; 38 -> 39 [object=6, label=3]; 38 -> 35 [object=6, label=2]; 35 -> 39 [object=6, label=2]; 38 -> 22 [object=6, label=55]; 22 -> 40 [object=6, label=55]; 40 -> 39 [object=6, label=53]; 40 -> 41 [object=6, label=2]; 41 -> 42 [object=6, label=2]; 42 -> 39 [object=6, label=2]; 39 -> 43 [object=6, label=60]; 43 -> 27 [object=6, label=60]; 27 -> 27 [object=6, label=71]; 27 -> 44 [object=6, label=60]; 43 -> 8 [object=6, label=60]; 8 -> 8 [object=6, label=3]; 8 -> 44 [object=6, label=60]; 43 -> 45 [object=6, label=60]; 45 -> 46 [object=6, label=155]; 46 -> 45 [object=6, label=155]; 45 -> 47 [object=6, label=39]; 47 -> 45 [object=6, label=39]; 45 -> 44 [object=6, label=60]; 44 -> 37 [object=6, label=60]; 48 -> 46 [object=7, label=38]; 46 -> 46 [object=7, label=117]; 46 -> 49 [object=7, label=38];}"""

    # big graphs
#    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateCreditMemo):4093"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_EXCLUSIVE_CHOICE"]; 5[label="BPMN_TASK(ClearCreditMemo):328090"]; 1[label="BPMN_END"]; 6[label="BPMN_START"]; 9[label="BPMN_EXCLUSIVE_CHOICE"]; 10[label="BPMN_EXCLUSIVE_CHOICE"]; 11[label="BPMN_TASK(PostCustomerInvoice):501165"]; 14[label="BPMN_PARALLEL"]; 16[label="BPMN_EXCLUSIVE_CHOICE"]; 17[label="BPMN_EXCLUSIVE_CHOICE"]; 18[label="BPMN_TASK(ChangePostedCustomerInvoice):35218"]; 15[label="BPMN_PARALLEL"]; 19[label="BPMN_EXCLUSIVE_CHOICE"]; 20[label="BPMN_EXCLUSIVE_CHOICE"]; 21[label="BPMN_TASK(CashDiscountDueDatePassed):24627"]; 22[label="BPMN_EXCLUSIVE_CHOICE"]; 23[label="BPMN_EXCLUSIVE_CHOICE"]; 24[label="BPMN_TASK(ClearCustomerInvoice):501278"]; 13[label="BPMN_PARALLEL"]; 25[label="BPMN_EXCLUSIVE_CHOICE"]; 26[label="BPMN_TASK(RemoveDunningBlock):20594"]; 27[label="BPMN_EXCLUSIVE_CHOICE"]; 28[label="BPMN_EXCLUSIVE_CHOICE"]; 29[label="BPMN_TASK(DueDatePassed):501165"]; 30[label="BPMN_PARALLEL"]; 32[label="BPMN_EXCLUSIVE_CHOICE"]; 33[label="BPMN_EXCLUSIVE_CHOICE"]; 34[label="BPMN_TASK(RemoveDunningNotices):58"]; 31[label="BPMN_PARALLEL"]; 35[label="BPMN_EXCLUSIVE_CHOICE"]; 36[label="BPMN_TASK(CreateDunningNoticesLevel2):334"]; 37[label="BPMN_EXCLUSIVE_CHOICE"]; 38[label="BPMN_EXCLUSIVE_CHOICE"]; 39[label="BPMN_TASK(CreateDunningNoticesLevel3):74"]; 12[label="BPMN_PARALLEL"]; 40[label="BPMN_EXCLUSIVE_CHOICE"]; 41[label="BPMN_TASK(CreateDunningNoticesLevel1):3216"]; 42[label="BPMN_EXCLUSIVE_CHOICE"]; 43[label="BPMN_TASK(SetDunningBlock):334"]; 44[label="BPMN_EXCLUSIVE_CHOICE"]; 8[label="BPMN_EXCLUSIVE_CHOICE"]; 45[label="BPMN_TASK(SetBillingBlock):374"]; 46[label="BPMN_TASK(CreateCustomerInvoice):688953"]; 47[label="BPMN_TASK(ReleaseCreditHold):143048"]; 48[label="BPMN_TASK(RemoveBillingBlock):465"]; 49[label="BPMN_TASK(SetCreditHold):13564"]; 50[label="BPMN_TASK(EnterInvoiceWithoutSalesOrder):25"]; 7[label="BPMN_END"]; 51[label="BPMN_START"]; 53[label="BPMN_PARALLEL"]; 55[label="BPMN_EXCLUSIVE_CHOICE"]; 56[label="BPMN_TASK(AddDeliveryItems):543735"]; 54[label="BPMN_PARALLEL"]; 57[label="BPMN_TASK(CreateDelivery):860495"]; 58[label="BPMN_EXCLUSIVE_CHOICE"]; 52[label="BPMN_END"]; 59[label="BPMN_START"]; 61[label="BPMN_PARALLEL"]; 62[label="BPMN_PARALLEL"]; 63[label="BPMN_EXCLUSIVE_CHOICE"]; 64[label="BPMN_EXCLUSIVE_CHOICE"]; 65[label="BPMN_EXCLUSIVE_CHOICE"]; 60[label="BPMN_END"]; 66[label="BPMN_START"]; 67[label="BPMN_END"]; 68[label="BPMN_START"]; 70[label="BPMN_TASK(CreateSalesQuotation):438"]; 71[label="BPMN_EXCLUSIVE_CHOICE"]; 72[label="BPMN_TASK(RejectSalesQuotation):665"]; 73[label="BPMN_TASK(CreateSalesOrder):698719"]; 74[label="BPMN_TASK(ChangeSalesQuotation):429"]; 75[label="BPMN_TASK(AddSalesQuotationItems):4"]; 69[label="BPMN_END"]; 76[label="BPMN_START"]; 78[label="BPMN_EXCLUSIVE_CHOICE"]; 80[label="BPMN_PARALLEL"]; 82[label="BPMN_EXCLUSIVE_CHOICE"]; 83[label="BPMN_EXCLUSIVE_CHOICE"]; 84[label="BPMN_PARALLEL"]; 85[label="BPMN_EXCLUSIVE_CHOICE"]; 86[label="BPMN_EXCLUSIVE_CHOICE"]; 81[label="BPMN_PARALLEL"]; 87[label="BPMN_EXCLUSIVE_CHOICE"]; 88[label="BPMN_TASK(ChangeSalesOrder):154735"]; 89[label="BPMN_TASK(SetDeliveryBlock):24002"]; 90[label="BPMN_TASK(RemoveDeliveryBlock):766880"]; 91[label="BPMN_TASK(PassCredit):15068"]; 92[label="BPMN_TASK(AddSalesOrderItems):16742"]; 93[label="BPMN_EXCLUSIVE_CHOICE"]; 79[label="BPMN_EXCLUSIVE_CHOICE"]; 94[label="BPMN_TASK(MassUpdateSalesOrder):211"]; 77[label="BPMN_END"]; 95[label="BPMN_START"]; 97[label="BPMN_PARALLEL"]; 99[label="BPMN_EXCLUSIVE_CHOICE"]; 100[label="BPMN_EXCLUSIVE_CHOICE"]; 101[label="BPMN_PARALLEL"]; 102[label="BPMN_EXCLUSIVE_CHOICE"]; 103[label="BPMN_EXCLUSIVE_CHOICE"]; 104[label="BPMN_TASK(SetRejectionReason):228098"]; 105[label="BPMN_TASK(ChangeSalesOrderScheduleLine):2064813"]; 106[label="BPMN_TASK(ChangeSalesOrderItem):190696"]; 107[label="BPMN_TASK(CancelRejectionReason):24280"]; 98[label="BPMN_PARALLEL"]; 108[label="BPMN_EXCLUSIVE_CHOICE"]; 109[label="BPMN_EXCLUSIVE_CHOICE"]; 96[label="BPMN_END"]; 110[label="BPMN_START"]; 112[label="BPMN_PARALLEL"]; 114[label="BPMN_EXCLUSIVE_CHOICE"]; 113[label="BPMN_PARALLEL"]; 111[label="BPMN_END"]; 0 -> 2 [object=0, label=4093]; 2 -> 3 [object=0, label=4093]; 3 -> 4 [object=0, label=896]; 3 -> 5 [object=0, label=3197]; 5 -> 4 [object=0, label=3197]; 4 -> 1 [object=0, label=4093]; 6 -> 9 [object=1, label=688953]; 9 -> 10 [object=1, label=839953]; 9 -> 11 [object=1, label=501165]; 11 -> 10 [object=1, label=501165]; 10 -> 14 [object=1, label=1341118]; 14 -> 16 [object=1, label=1341118]; 16 -> 17 [object=1, label=1305900]; 16 -> 18 [object=1, label=35218]; 18 -> 17 [object=1, label=35218]; 17 -> 15 [object=1, label=1341118]; 14 -> 19 [object=1, label=1341118]; 19 -> 20 [object=1, label=1316517]; 19 -> 21 [object=1, label=24601]; 21 -> 20 [object=1, label=24601]; 20 -> 15 [object=1, label=1341118]; 15 -> 22 [object=1, label=1341118]; 22 -> 23 [object=1, label=1007852]; 22 -> 24 [object=1, label=333266]; 24 -> 23 [object=1, label=333266]; 23 -> 13 [object=1, label=1341118]; 14 -> 25 [object=1, label=1341118]; 25 -> 26 [object=1, label=20594]; 26 -> 25 [object=1, label=20594]; 25 -> 13 [object=1, label=1341118]; 13 -> 27 [object=1, label=1341118]; 27 -> 28 [object=1, label=849185]; 27 -> 29 [object=1, label=491933]; 29 -> 28 [object=1, label=491933]; 28 -> 30 [object=1, label=1341118]; 30 -> 32 [object=1, label=1341118]; 32 -> 33 [object=1, label=1341060]; 32 -> 34 [object=1, label=58]; 34 -> 33 [object=1, label=58]; 33 -> 31 [object=1, label=1341118]; 30 -> 35 [object=1, label=1341118]; 35 -> 36 [object=1, label=334]; 36 -> 35 [object=1, label=334]; 35 -> 31 [object=1, label=1341118]; 31 -> 37 [object=1, label=1341118]; 37 -> 38 [object=1, label=1341044]; 37 -> 39 [object=1, label=74]; 39 -> 38 [object=1, label=74]; 38 -> 12 [object=1, label=1341118]; 30 -> 40 [object=1, label=1341118]; 40 -> 41 [object=1, label=3216]; 41 -> 40 [object=1, label=3216]; 40 -> 12 [object=1, label=1341118]; 14 -> 42 [object=1, label=1341118]; 42 -> 43 [object=1, label=334]; 43 -> 42 [object=1, label=334]; 42 -> 12 [object=1, label=1341118]; 12 -> 44 [object=1, label=1341118]; 44 -> 8 [object=1, label=1341039]; 44 -> 45 [object=1, label=79]; 45 -> 8 [object=1, label=79]; 9 -> 46 [object=1, label=688953]; 46 -> 8 [object=1, label=688953]; 9 -> 47 [object=1, label=11000]; 47 -> 8 [object=1, label=11000]; 8 -> 9 [object=1, label=1350515]; 8 -> 48 [object=1, label=304]; 48 -> 9 [object=1, label=304]; 8 -> 49 [object=1, label=1285]; 49 -> 49 [object=1, label=69]; 49 -> 9 [object=1, label=1285]; 8 -> 50 [object=1, label=14]; 50 -> 50 [object=1, label=11]; 50 -> 9 [object=1, label=14]; 8 -> 7 [object=1, label=688953]; 51 -> 53 [object=2, label=860495]; 53 -> 55 [object=2, label=860495]; 55 -> 56 [object=2, label=610537]; 56 -> 55 [object=2, label=610537]; 55 -> 54 [object=2, label=860495]; 53 -> 57 [object=2, label=860495]; 57 -> 58 [object=2, label=860495]; 58 -> 46 [object=2, label=381431]; 46 -> 58 [object=2, label=381431]; 58 -> 54 [object=2, label=860495]; 54 -> 52 [object=2, label=860495]; 59 -> 61 [object=3, label=4703594]; 61 -> 57 [object=3, label=4703594]; 57 -> 62 [object=3, label=4703594]; 61 -> 63 [object=3, label=4703594]; 63 -> 46 [object=3, label=1817658]; 46 -> 63 [object=3, label=1817658]; 63 -> 62 [object=3, label=4703594]; 61 -> 64 [object=3, label=4703594]; 64 -> 65 [object=3, label=4091191]; 64 -> 56 [object=3, label=612403]; 56 -> 65 [object=3, label=612403]; 65 -> 62 [object=3, label=4703594]; 62 -> 60 [object=3, label=4703594]; 66 -> 67 [object=4, label=0]; 68 -> 70 [object=5, label=438]; 70 -> 71 [object=5, label=438]; 71 -> 72 [object=5, label=665]; 72 -> 71 [object=5, label=665]; 71 -> 73 [object=5, label=3856]; 73 -> 71 [object=5, label=3856]; 71 -> 74 [object=5, label=429]; 74 -> 71 [object=5, label=429]; 71 -> 75 [object=5, label=19]; 75 -> 71 [object=5, label=19]; 71 -> 69 [object=5, label=438]; 76 -> 78 [object=6, label=698719]; 78 -> 80 [object=6, label=699027]; 80 -> 82 [object=6, label=699027]; 82 -> 83 [object=6, label=308]; 82 -> 73 [object=6, label=698719]; 73 -> 83 [object=6, label=698719]; 83 -> 84 [object=6, label=699027]; 84 -> 85 [object=6, label=699027]; 85 -> 45 [object=6, label=374]; 45 -> 85 [object=6, label=374]; 85 -> 48 [object=6, label=465]; 48 -> 85 [object=6, label=465]; 85 -> 86 [object=6, label=699027]; 86 -> 49 [object=6, label=13564]; 49 -> 86 [object=6, label=13564]; 86 -> 81 [object=6, label=699027]; 84 -> 87 [object=6, label=699027]; 87 -> 88 [object=6, label=154735]; 88 -> 87 [object=6, label=154735]; 87 -> 89 [object=6, label=24002]; 89 -> 87 [object=6, label=24002]; 87 -> 90 [object=6, label=766880]; 90 -> 87 [object=6, label=766880]; 87 -> 47 [object=6, label=143048]; 47 -> 87 [object=6, label=143048]; 87 -> 91 [object=6, label=15068]; 91 -> 87 [object=6, label=15068]; 87 -> 92 [object=6, label=16742]; 92 -> 87 [object=6, label=16742]; 87 -> 81 [object=6, label=699027]; 80 -> 93 [object=6, label=699027]; 93 -> 57 [object=6, label=660369]; 57 -> 93 [object=6, label=660369]; 93 -> 81 [object=6, label=699027]; 81 -> 79 [object=6, label=699027]; 79 -> 94 [object=6, label=308]; 94 -> 94 [object=6, label=278]; 94 -> 78 [object=6, label=308]; 79 -> 77 [object=6, label=698719]; 95 -> 97 [object=7, label=2890715]; 97 -> 99 [object=7, label=2890715]; 99 -> 100 [object=7, label=26474]; 99 -> 73 [object=7, label=2864241]; 73 -> 100 [object=7, label=2864241]; 100 -> 101 [object=7, label=2890715]; 101 -> 102 [object=7, label=2890715]; 102 -> 103 [object=7, label=2857962]; 102 -> 92 [object=7, label=32753]; 92 -> 103 [object=7, label=32753]; 103 -> 104 [object=7, label=227795]; 104 -> 103 [object=7, label=227795]; 103 -> 105 [object=7, label=2063953]; 105 -> 103 [object=7, label=2063953]; 103 -> 106 [object=7, label=190600]; 106 -> 103 [object=7, label=190600]; 103 -> 107 [object=7, label=24228]; 107 -> 103 [object=7, label=24228]; 103 -> 98 [object=7, label=2890715]; 101 -> 108 [object=7, label=2890715]; 108 -> 94 [object=7, label=70517]; 94 -> 108 [object=7, label=70517]; 108 -> 88 [object=7, label=718467]; 88 -> 108 [object=7, label=718467]; 108 -> 98 [object=7, label=2890715]; 97 -> 109 [object=7, label=2890715]; 109 -> 57 [object=7, label=2534923]; 57 -> 109 [object=7, label=2534923]; 109 -> 98 [object=7, label=2890715]; 98 -> 96 [object=7, label=2890715]; 110 -> 112 [object=8, label=516593]; 112 -> 114 [object=8, label=516593]; 114 -> 94 [object=8, label=94114]; 94 -> 114 [object=8, label=94114]; 114 -> 113 [object=8, label=516593]; 112 -> 105 [object=8, label=516593]; 105 -> 105 [object=8, label=1548220]; 105 -> 113 [object=8, label=516593]; 113 -> 111 [object=8, label=516593];}"""
#    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateDelivery):387740"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 7[label="BPMN_EXCLUSIVE_CHOICE"]; 6[label="BPMN_TASK(AddDeliveryItems):3712"]; 8[label="BPMN_TASK(ExecutePicking):142741"]; 9[label="BPMN_TASK(CreateProFormaInvoice):93700"]; 5[label="BPMN_EXCLUSIVE_CHOICE"]; 10[label="BPMN_TASK(SendOrderConfirmation):112776"]; 11[label="BPMN_PARALLEL"]; 13[label="BPMN_EXCLUSIVE_CHOICE"]; 14[label="BPMN_EXCLUSIVE_CHOICE"]; 15[label="BPMN_TASK(PostGoodsIssue):916793"]; 12[label="BPMN_PARALLEL"]; 16[label="BPMN_EXCLUSIVE_CHOICE"]; 17[label="BPMN_TASK(PassCredit):1178"]; 18[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_EXCLUSIVE_CHOICE"]; 19[label="BPMN_TASK(CreateCustomerInvoice):324013"]; 20[label="BPMN_TASK(SplitOutboundDelivery):82"]; 1[label="BPMN_END"]; 21[label="BPMN_START"]; 23[label="BPMN_PARALLEL"]; 25[label="BPMN_EXCLUSIVE_CHOICE"]; 26[label="BPMN_EXCLUSIVE_CHOICE"]; 27[label="BPMN_EXCLUSIVE_CHOICE"]; 28[label="BPMN_TASK(AddSalesQuotationItems):3968"]; 29[label="BPMN_TASK(ApproveSalesQuotationItem):407505"]; 24[label="BPMN_PARALLEL"]; 30[label="BPMN_TASK(CreateSalesQuotation):136653"]; 31[label="BPMN_EXCLUSIVE_CHOICE"]; 32[label="BPMN_TASK(RejectSalesQuotation):455"]; 33[label="BPMN_TASK(ChangeSalesQuotation):24970"]; 34[label="BPMN_EXCLUSIVE_CHOICE"]; 35[label="BPMN_TASK(CreateSalesOrder):117970"]; 22[label="BPMN_END"]; 36[label="BPMN_START"]; 38[label="BPMN_PARALLEL"]; 40[label="BPMN_EXCLUSIVE_CHOICE"]; 41[label="BPMN_EXCLUSIVE_CHOICE"]; 42[label="BPMN_TASK(ApproveSalesOrder):9127"]; 43[label="BPMN_EXCLUSIVE_CHOICE"]; 44[label="BPMN_TASK(SetDeliveryBlock):221"]; 45[label="BPMN_PARALLEL"]; 46[label="BPMN_EXCLUSIVE_CHOICE"]; 48[label="BPMN_PARALLEL"]; 50[label="BPMN_EXCLUSIVE_CHOICE"]; 52[label="BPMN_EXCLUSIVE_CHOICE"]; 53[label="BPMN_TASK(ChangeSalesOrder):4995"]; 54[label="BPMN_EXCLUSIVE_CHOICE"]; 55[label="BPMN_TASK(ReleaseCreditHold):4222"]; 51[label="BPMN_EXCLUSIVE_CHOICE"]; 56[label="BPMN_TASK(RemoveDeliveryBlock):370"]; 49[label="BPMN_PARALLEL"]; 57[label="BPMN_EXCLUSIVE_CHOICE"]; 47[label="BPMN_EXCLUSIVE_CHOICE"]; 58[label="BPMN_TASK(SetBillingBlock):134"]; 39[label="BPMN_PARALLEL"]; 59[label="BPMN_EXCLUSIVE_CHOICE"]; 60[label="BPMN_TASK(RemoveBillingBlock):1441"]; 61[label="BPMN_EXCLUSIVE_CHOICE"]; 62[label="BPMN_EXCLUSIVE_CHOICE"]; 37[label="BPMN_END"]; 63[label="BPMN_START"]; 65[label="BPMN_EXCLUSIVE_CHOICE"]; 66[label="BPMN_EXCLUSIVE_CHOICE"]; 67[label="BPMN_TASK(ApproveSalesOrderItem):3260"]; 68[label="BPMN_PARALLEL"]; 71[label="BPMN_EXCLUSIVE_CHOICE"]; 72[label="BPMN_EXCLUSIVE_CHOICE"]; 70[label="BPMN_PARALLEL"]; 73[label="BPMN_EXCLUSIVE_CHOICE"]; 74[label="BPMN_TASK(CreateSalesOrderScheduleLine):1452486"]; 75[label="BPMN_TASK(ChangeSalesOrderScheduleLine):1382888"]; 76[label="BPMN_EXCLUSIVE_CHOICE"]; 77[label="BPMN_EXCLUSIVE_CHOICE"]; 78[label="BPMN_TASK(SetRejectionReason):1079"]; 69[label="BPMN_PARALLEL"]; 79[label="BPMN_EXCLUSIVE_CHOICE"]; 64[label="BPMN_END"]; 0 -> 2 [object=0, label=936580]; 2 -> 3 [object=0, label=936580]; 3 -> 7 [object=0, label=919253]; 3 -> 6 [object=0, label=16980]; 6 -> 7 [object=0, label=16980]; 7 -> 8 [object=0, label=324905]; 8 -> 7 [object=0, label=324905]; 7 -> 9 [object=0, label=238153]; 9 -> 7 [object=0, label=238153]; 7 -> 5 [object=0, label=936233]; 3 -> 10 [object=0, label=283]; 10 -> 10 [object=0, label=283]; 10 -> 5 [object=0, label=283]; 5 -> 11 [object=0, label=936516]; 11 -> 13 [object=0, label=936516]; 13 -> 14 [object=0, label=934776]; 13 -> 15 [object=0, label=1740]; 15 -> 14 [object=0, label=1740]; 14 -> 12 [object=0, label=936516]; 11 -> 16 [object=0, label=936516]; 16 -> 17 [object=0, label=2643]; 17 -> 16 [object=0, label=2643]; 16 -> 12 [object=0, label=936516]; 12 -> 18 [object=0, label=936516]; 18 -> 4 [object=0, label=936357]; 18 -> 19 [object=0, label=159]; 19 -> 4 [object=0, label=159]; 3 -> 20 [object=0, label=64]; 20 -> 4 [object=0, label=64]; 4 -> 1 [object=0, label=936580]; 21 -> 23 [object=1, label=407503]; 23 -> 25 [object=1, label=407503]; 25 -> 26 [object=1, label=2]; 25 -> 27 [object=1, label=382836]; 25 -> 28 [object=1, label=24665]; 28 -> 27 [object=1, label=24665]; 27 -> 29 [object=1, label=407501]; 29 -> 26 [object=1, label=407501]; 26 -> 24 [object=1, label=407503]; 23 -> 30 [object=1, label=407503]; 30 -> 24 [object=1, label=407503]; 24 -> 31 [object=1, label=407503]; 31 -> 32 [object=1, label=451]; 32 -> 31 [object=1, label=451]; 31 -> 33 [object=1, label=24647]; 33 -> 31 [object=1, label=24647]; 31 -> 34 [object=1, label=407503]; 34 -> 35 [object=1, label=224]; 35 -> 34 [object=1, label=224]; 34 -> 22 [object=1, label=407503]; 36 -> 38 [object=2, label=117241]; 38 -> 40 [object=2, label=117241]; 40 -> 41 [object=2, label=108828]; 40 -> 42 [object=2, label=8413]; 42 -> 41 [object=2, label=8413]; 41 -> 43 [object=2, label=117203]; 41 -> 44 [object=2, label=38]; 44 -> 43 [object=2, label=38]; 43 -> 45 [object=2, label=117241]; 45 -> 46 [object=2, label=117241]; 46 -> 48 [object=2, label=117191]; 48 -> 50 [object=2, label=117191]; 50 -> 52 [object=2, label=117120]; 52 -> 53 [object=2, label=2267]; 53 -> 52 [object=2, label=2267]; 52 -> 54 [object=2, label=2637]; 52 -> 17 [object=2, label=259]; 17 -> 54 [object=2, label=259]; 54 -> 55 [object=2, label=2896]; 55 -> 52 [object=2, label=2896]; 52 -> 51 [object=2, label=117120]; 50 -> 56 [object=2, label=71]; 56 -> 51 [object=2, label=71]; 51 -> 49 [object=2, label=117191]; 48 -> 57 [object=2, label=117191]; 57 -> 10 [object=2, label=112080]; 10 -> 57 [object=2, label=112080]; 57 -> 49 [object=2, label=117191]; 49 -> 47 [object=2, label=117191]; 46 -> 58 [object=2, label=50]; 58 -> 47 [object=2, label=50]; 47 -> 39 [object=2, label=117241]; 45 -> 59 [object=2, label=117241]; 59 -> 60 [object=2, label=1206]; 60 -> 59 [object=2, label=1206]; 59 -> 39 [object=2, label=117241]; 38 -> 35 [object=2, label=117241]; 35 -> 39 [object=2, label=117241]; 39 -> 61 [object=2, label=117241]; 61 -> 62 [object=2, label=117208]; 61 -> 2 [object=2, label=33]; 2 -> 62 [object=2, label=33]; 62 -> 37 [object=2, label=117241]; 63 -> 65 [object=3, label=3012]; 65 -> 66 [object=3, label=228]; 65 -> 35 [object=3, label=2784]; 35 -> 66 [object=3, label=2784]; 66 -> 67 [object=3, label=3012]; 67 -> 68 [object=3, label=3012]; 68 -> 71 [object=3, label=3012]; 71 -> 72 [object=3, label=1156]; 71 -> 2 [object=3, label=1856]; 2 -> 72 [object=3, label=1856]; 72 -> 70 [object=3, label=3012]; 68 -> 73 [object=3, label=3012]; 73 -> 74 [object=3, label=250]; 74 -> 73 [object=3, label=250]; 73 -> 75 [object=3, label=500]; 75 -> 73 [object=3, label=500]; 73 -> 70 [object=3, label=3012]; 70 -> 76 [object=3, label=3012]; 76 -> 77 [object=3, label=1937]; 76 -> 9 [object=3, label=36]; 9 -> 9 [object=3, label=36]; 9 -> 77 [object=3, label=36]; 76 -> 78 [object=3, label=1039]; 78 -> 77 [object=3, label=1039]; 77 -> 69 [object=3, label=3012]; 68 -> 79 [object=3, label=3012]; 79 -> 53 [object=3, label=7306]; 53 -> 79 [object=3, label=7306]; 79 -> 69 [object=3, label=3012]; 69 -> 64 [object=3, label=3012];}"""

    graph = """digraph { 0[label="BPMN_START"]; 2[label="BPMN_TASK(CreateCustomerInvoice):135138"]; 3[label="BPMN_EXCLUSIVE_CHOICE"]; 4[label="BPMN_TASK(SendOverdueNotice):78174"]; 5[label="BPMN_TASK(ClearCustomerInvoice):135138"]; 1[label="BPMN_END"]; 6[label="BPMN_START"]; 8[label="BPMN_TASK(CreateDelivery):135138"]; 9[label="BPMN_TASK(ExecutePicking):135138"]; 10[label="BPMN_EXCLUSIVE_CHOICE"]; 11[label="BPMN_EXCLUSIVE_CHOICE"]; 12[label="BPMN_TASK(InsufficientMaterialFound):28381"]; 13[label="BPMN_TASK(PostGoodsIssue):135138"]; 7[label="BPMN_END"]; 14[label="BPMN_START"]; 16[label="BPMN_TASK(PostGoodsReceipt):135138"]; 17[label="BPMN_EXCLUSIVE_CHOICE"]; 18[label="BPMN_EXCLUSIVE_CHOICE"]; 19[label="BPMN_TASK(ReceiveVendorInvoice):135138"]; 15[label="BPMN_END"]; 20[label="BPMN_START"]; 22[label="BPMN_TASK(CreatePurchaseOrder):135138"]; 23[label="BPMN_EXCLUSIVE_CHOICE"]; 25[label="BPMN_PARALLEL"]; 27[label="BPMN_TASK(ApprovePurchaseOrder):130896"]; 28[label="BPMN_TASK(SendPurchaseOrder):130896"]; 26[label="BPMN_PARALLEL"]; 29[label="BPMN_EXCLUSIVE_CHOICE"]; 30[label="BPMN_TASK(ChangePrice):14948"]; 31[label="BPMN_EXCLUSIVE_CHOICE"]; 24[label="BPMN_EXCLUSIVE_CHOICE"]; 32[label="BPMN_TASK(SendDeliveryOverdueNotice):28381"]; 33[label="BPMN_TASK(ChangeQuantity):19897"]; 21[label="BPMN_END"]; 34[label="BPMN_START"]; 36[label="BPMN_TASK(CreateQuotation):135138"]; 37[label="BPMN_TASK(ApproveQuotation):137966"]; 38[label="BPMN_TASK(CreateSalesOrder):135138"]; 35[label="BPMN_END"]; 39[label="BPMN_START"]; 41[label="BPMN_EXCLUSIVE_CHOICE"]; 42[label="BPMN_EXCLUSIVE_CHOICE"]; 43[label="BPMN_PARALLEL"]; 45[label="BPMN_EXCLUSIVE_CHOICE"]; 46[label="BPMN_EXCLUSIVE_CHOICE"]; 47[label="BPMN_TASK(ChangeShipTo):4242"]; 48[label="BPMN_TASK(RemoveDeliveryBlock):46864"]; 44[label="BPMN_PARALLEL"]; 49[label="BPMN_TASK(RemoveCreditBlock):48278"]; 50[label="BPMN_TASK(ChangeSoldTo):1414"]; 40[label="BPMN_END"]; 51[label="BPMN_START"]; 53[label="BPMN_EXCLUSIVE_CHOICE"]; 54[label="BPMN_EXCLUSIVE_CHOICE"]; 55[label="BPMN_TASK(ChangeSalesOrderItem):81305"]; 52[label="BPMN_END"]; 0 -> 2 [object=0, label=135138]; 2 -> 3 [object=0, label=135138]; 3 -> 4 [object=0, label=78174]; 4 -> 3 [object=0, label=78174]; 3 -> 5 [object=0, label=135138]; 5 -> 1 [object=0, label=135138]; 6 -> 8 [object=1, label=1086053]; 8 -> 9 [object=1, label=1086053]; 9 -> 10 [object=1, label=1086053]; 10 -> 11 [object=1, label=856884]; 10 -> 12 [object=1, label=229169]; 12 -> 11 [object=1, label=229169]; 11 -> 13 [object=1, label=1086053]; 13 -> 2 [object=1, label=1086053]; 2 -> 7 [object=1, label=1086053]; 14 -> 16 [object=2, label=605697]; 16 -> 9 [object=2, label=605697]; 9 -> 17 [object=2, label=605697]; 17 -> 18 [object=2, label=479346]; 17 -> 12 [object=2, label=126351]; 12 -> 18 [object=2, label=126351]; 18 -> 13 [object=2, label=605697]; 13 -> 19 [object=2, label=605697]; 19 -> 15 [object=2, label=605697]; 20 -> 22 [object=3, label=341077]; 22 -> 23 [object=3, label=341077]; 23 -> 25 [object=3, label=330169]; 25 -> 27 [object=3, label=330169]; 27 -> 28 [object=3, label=330169]; 28 -> 26 [object=3, label=330169]; 25 -> 29 [object=3, label=330169]; 29 -> 30 [object=3, label=10706]; 30 -> 29 [object=3, label=10706]; 29 -> 26 [object=3, label=330169]; 26 -> 31 [object=3, label=330169]; 31 -> 24 [object=3, label=257954]; 31 -> 32 [object=3, label=72215]; 32 -> 24 [object=3, label=72215]; 23 -> 33 [object=3, label=10908]; 33 -> 33 [object=3, label=3333]; 33 -> 24 [object=3, label=10908]; 24 -> 16 [object=3, label=341077]; 16 -> 19 [object=3, label=341077]; 19 -> 21 [object=3, label=341077]; 34 -> 36 [object=4, label=135138]; 36 -> 37 [object=4, label=135138]; 37 -> 37 [object=4, label=2828]; 37 -> 38 [object=4, label=135138]; 38 -> 35 [object=4, label=135138]; 39 -> 38 [object=5, label=135138]; 38 -> 41 [object=5, label=135138]; 41 -> 42 [object=5, label=86860]; 41 -> 43 [object=5, label=46864]; 43 -> 45 [object=5, label=46864]; 45 -> 46 [object=5, label=42622]; 45 -> 47 [object=5, label=4242]; 47 -> 46 [object=5, label=4242]; 46 -> 48 [object=5, label=46864]; 48 -> 44 [object=5, label=46864]; 43 -> 49 [object=5, label=46864]; 49 -> 49 [object=5, label=1414]; 49 -> 44 [object=5, label=46864]; 44 -> 42 [object=5, label=46864]; 41 -> 50 [object=5, label=1414]; 50 -> 42 [object=5, label=1414]; 42 -> 8 [object=5, label=135138]; 8 -> 2 [object=5, label=135138]; 2 -> 40 [object=5, label=135138]; 51 -> 38 [object=6, label=610848]; 38 -> 53 [object=6, label=610848]; 53 -> 54 [object=6, label=532876]; 53 -> 55 [object=6, label=77972]; 55 -> 54 [object=6, label=77972]; 54 -> 8 [object=6, label=610848]; 8 -> 2 [object=6, label=610848]; 2 -> 52 [object=6, label=610848];}"""

    G, object_types = create_graph(graph)
    graphs = {}
    traces = {}
    for ot in object_types:

        graph = flatten_graph(G, ot)
        graphs[ot] = graph

#        if ot == "0":
#            to_dot_format(graph, "output/0.dot")

        simulation = Simulation(graph, max_trace_length=50, max_cycles=3, minimize_for_node_coverage=True)
        simulation.start_simulation()
        traces[ot] = simulation.get_activity_sequence_representation(ignore_self_loops=False)
        print(ot)
        print("Uncovered nodes:", simulation.get_uncovered_nodes_coverage())
        print("Number of traces:", len(traces[ot]))
