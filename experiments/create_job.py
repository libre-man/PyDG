#!/usr/bin/env python3.6
import ast
import csv
import fnmatch
import itertools
import os
import typing as t
from argparse import ArgumentParser

import networkx as nx

from codegra_plag import cdg, graph, pdg

THRESHOLD = 15
GAMMA = 0.8
MIN_SIZE = 15

__CACHE = {}


def to_mcgregor(graph) -> t.List[str]:
    """Export a networkx graph to the format required by `run.bash`.
    """
    mapping = {}
    out = []
    for idx, node in enumerate(graph.nodes):
        mapping[node] = idx
        out.append(str(idx))
    for head, tail in graph.edges:
        out.append('{},{}'.format(mapping[head], mapping[tail]))
    return out


def find_enter(
        graph: graph.Graph[cdg.CDGNode]) -> t.Optional[cdg.CDGRegionNode]:
    """Find the enter node of a graph.
    """
    for node in graph.nodes:
        if isinstance(node, cdg.CDGRegionNode
                      ) and node.type == cdg.CDGRegionNodeType.enter:
            return node
    return None


def find_files(directory, pattern):
    """Find all files in a directory that comply to the specify pattern.
    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                if os.path.isfile(filename):
                    yield filename


def get_pdgs(d, split):
    """Get all PDGs in a directory ``d`` and maybe split them.

    :param d: The directory to search in.
    :param split: Split the PDGs.
    :returns: A ``Subgraph`` instance
    """
    pdgs = []
    for f_name in find_files(d, "*.py"):
        try:
            with open(f_name, 'r') as f:
                tree = ast.parse(f.read())
        except Exception as e:
            # print('Invalid syntax in', f, e)
            continue
        else:
            cur_pdg = pdg.create_pdg(tree)
            cur_pdg.remove_useless_nodes()
            assert not cur_pdg.edges
            res = SubGraphs()
            for cur_subgraph in cur_pdg.subgraphs:
                starter = cur_subgraph.to_networkx().to_undirected()
                if not split:
                    res.subgraphs.append(starter)
                    continue
                todo = [starter]
                first = True
                while todo:
                    g = todo.pop()
                    if len(g.nodes) >= MIN_SIZE:
                        res.subgraphs.append(g.copy())
                        cut = nx.minimum_edge_cut(g)
                        g.remove_edges_from(cut)
                        todo.extend(
                            g.subgraph(c).copy()
                            for c in sorted(
                                nx.connected_components(g),
                                key=len,
                                reverse=True))
                        if not first and len(todo[-1].nodes) < 2 and len(
                                todo[-2].nodes) >= MIN_SIZE:
                            res.subgraphs.pop()
                    first = False
            pdgs.append((f_name, res))

    return pdgs


class SubGraphs:
    def __init__(self):
        self.subgraphs = []


def create_csv(output_name, dirs, use_mincut):
    cache_1 = {}
    cache_2 = {}
    with open(output_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'student1', 'student2', 'file1', 'file2', 'func1', 'func2',
            'graph1', 'graph2'
        ])
        for d1, d2 in itertools.product(dirs, repeat=2):
            print(d1, d2)
            if d1 == d2:
                continue
            if d1 not in cache_1:
                cache_1[d1] = get_pdgs(d1, False)
            if d2 not in cache_2:
                cache_2[d2] = get_pdgs(d2, bool(use_mincut))

            pdgs1 = cache_1[d1]
            pdgs2 = cache_2[d2]

            for (f1, pdg1), (f2, pdg2) in itertools.product(pdgs1, pdgs2):
                for sub1, sub2 in itertools.product(pdg1.subgraphs,
                                                    pdg2.subgraphs):
                    if len(sub1.nodes) < len(sub2.nodes):
                        continue
                    if len(sub2.nodes) < MIN_SIZE:
                        continue
                    if len(sub1.edges) < 2 or len(sub2.edges) < 2:
                        continue

                    enter1 = find_enter(sub1)
                    enter2 = find_enter(sub2)
                    name1 = enter1.label if enter1 else 'UNKOWN'
                    name2 = enter2.label if enter2 else 'UNKOWN'

                    writer.writerow([
                        d1, d2, f1, f2, name1, name2,
                        to_mcgregor(sub1),
                        to_mcgregor(sub2)
                    ])


def main():
    argparser = ArgumentParser(description='Create jobs using CodeGra.plag')
    argparser.add_argument(
        'directories',
        nargs='+',
        help='The directories of the files that should be checked')
    argparser.add_argument(
        '-m',
        '--mincut',
        dest='mincut',
        action='store_true',
        default=False,
        help='Use mincut to process graphs')
    argparser.add_argument(
        '-o',
        '--output',
        default='/dev/stdout',
        dest='output',
        type=str,
        metavar='OUTPUT',
        help='The destination of the output')
    args = argparser.parse_args()
    create_csv(args.output, args.directories, args.mincut)

if __name__ == '__main__':
    main()
