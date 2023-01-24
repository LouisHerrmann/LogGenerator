import networkx as nx
from copy import deepcopy
import pydotplus


class IdFactory:
    def __init__(self):
        self.id = 999

    def get_new_id(self):
        self.id += 1
        return self.id


def create_graph(dot_graph):
    object_types = set()

    # remove newline characters since they can cause issues and be interpreted as nodes
    dot_graph = dot_graph.replace("\n", "")
    dot_graph = pydotplus.graph_from_dot_data(dot_graph)

    id_factory = IdFactory()
    id_mapping = {}

    graph = nx.DiGraph()
    for edge in dot_graph.get_edges():
        object_type = edge.get_attributes()["object"]
        object_types.add(object_type)
        s_id, d_id = edge.get_source(), edge.get_destination()
        for node_id in s_id, d_id:
            node = dot_graph.get_node(node_id)[0]

            if node_id not in id_mapping:
                new_id = id_factory.get_new_id()
                id_mapping[node_id] = new_id
                graph.add_node(new_id, object_types=set(), act_name=node.get_attributes()["label"])

            # annotate node with object types of ingoing / outgoing arcs
            related_object_types = graph.nodes.get(id_mapping[node_id])["object_types"].add(object_type)
            nx.set_node_attributes(graph, {node_id: {"object_types": related_object_types}})

        graph.add_edge(id_mapping[s_id], id_mapping[d_id], object_type=object_type)

    return graph, object_types


def flatten_graph(graph, ot):
    graph = deepcopy(graph)
    to_be_removed_nodes = []
    to_be_removed_edges = []

    for node_id in graph.nodes:
        if ot not in graph.nodes.get(node_id)["object_types"]:
            to_be_removed_nodes.append(node_id)

    for edge_id in graph.edges:
        if ot != graph.edges.get(edge_id)["object_type"]:
            to_be_removed_edges.append(edge_id)

    graph.remove_nodes_from(to_be_removed_nodes)
    graph.remove_edges_from(to_be_removed_edges)
    return graph


def remove_self_loops(graph):
    # for replay, we can simply remove self loops since activities with self loops can just be
    # duplicated in the trace later
    graph = deepcopy(graph)
    to_be_removed = []
    for edge in graph.edges:
        if edge[0] == edge[1]:
            to_be_removed.append(edge)

    graph.remove_edges_from(to_be_removed)
    return graph, [edge[0] for edge in to_be_removed]


def to_dot_format(graph, file_path):
    nx.drawing.nx_pydot.write_dot(graph, file_path)
