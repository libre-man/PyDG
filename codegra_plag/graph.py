"""This module implements a generic type safe graph.
"""
import collections
import typing as t
import uuid
from abc import ABCMeta

import graphviz  # type: ignore
import networkx as nx  # type: ignore

T = t.TypeVar('T', bound='Node')


class Edge(t.Generic[T]):
    def __init__(
        self,
        head: T,
        tail: T,
        label: t.Optional[str],
        dotted: bool=False,
    ) -> None:
        self.head = head
        self.tail = tail
        self.label = label
        self.dotted = dotted

    def __str__(self) -> str:
        if self.tail:
            return '{} -({})- {}'.format(self.head, self.label, self.tail)
        else:
            return '{} - {}'.format(self.head, self.tail)


class Node:
    def __init__(self, label: str) -> None:
        self.label = label

    def __str__(self) -> str:
        return self.label


class Graph(t.Generic[T]):
    def __init__(self) -> None:
        self.edges: t.List[Edge[T]] = list()
        self._nodes_set: t.Set[T] = set()
        self.nodes: t.List[T] = []
        self.subgraphs: t.List['Graph[T]'] = []

    def add_subgraph(self, subgraph: 'Graph[T]') -> None:
        self.subgraphs.append(subgraph)

    def add_edge(self, edge: Edge[T]) -> None:
        if edge.head not in self._nodes_set:
            self.nodes.append(edge.head)
            self._nodes_set.add(edge.head)
        if edge.tail not in self._nodes_set:
            self.nodes.append(edge.tail)
            self._nodes_set.add(edge.tail)
        self.edges.append(edge)

    def to_networkx(self) -> nx.DiGraph:
        """Export this graph to a directed networkx graph. This loses metadata.
        """
        res = nx.DiGraph()
        res.add_edges_from(list((edge.head, edge.tail) for edge in self.edges))

        for subgraph in self.subgraphs:
            res.add_edges_from(subgraph.to_networkx().edges())
        return res

    def to_dot(self) -> graphviz.Digraph:
        """Render this graph as a dot file.
        """
        graph = graphviz.Digraph()
        id_map: t.MutableMapping[Node, str] = collections.defaultdict(
            lambda: str(uuid.uuid4())
        )
        done: t.Set[Node] = set()

        for i, subgraph in enumerate([self] + self.subgraphs):
            with graph.subgraph(name='sub_{}'.format(i)) as dot:
                for node in subgraph.nodes:
                    dot.node(id_map[node], node.label)

                for edge in subgraph.edges:
                    dot.edge(
                        id_map[edge.head],
                        id_map[edge.tail],
                        label=edge.label,
                        style='dotted' if edge.dotted else 'solid'
                    )
        return graph

    def remove_useless_nodes(self) -> None:
        """Remove useless nodes by modifying this graph as described in
        "T. Schaper. Using program dependency graphs for plagiarism detection
        in python. 2018."
        """
        for sub in self.subgraphs:
            sub.remove_useless_nodes()
        removed = True
        while removed:
            mapping: t.MutableMapping[T, int] = collections.defaultdict(int)
            for edge in self.edges:
                mapping[edge.head] += 1
                mapping[edge.tail] += 1
            new_nodes = [n for n in self.nodes if mapping[n] > 1]
            removed = len(new_nodes) < len(self.nodes)
            self.nodes = new_nodes
            self.edges = [
                e for e in self.edges
                if mapping[e.head] > 1 and mapping[e.tail] > 1
            ]
