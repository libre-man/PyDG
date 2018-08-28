"""This module creates a PDG for python by combining a CFG and a DDG. Use the
``create_pdg`` function to do this.
"""
import ast
import typing as t

from . import cdg, ddg, simplify

PDG = t.NewType('PDG', cdg.CDG)


def create_pdg(tree: ast.Module) -> PDG:
    tree.body = simplify.simplify_stmts(tree.body)
    cur_cdg = cdg.create_cdg(tree)
    ddg_tree = ddg.create_ddg(tree)

    lookup: t.MutableMapping[t.Union[ast.stmt, ast.ExceptHandler],
                             t.Tuple[cdg.CDGNode, cdg.CDG]] = {}
    todo = [cur_cdg]
    while todo:
        cur = todo.pop()
        for node in cur.nodes:
            lookup[node.orig] = (node, cur)
        todo.extend(cur.subgraphs)

    for key, vals in ddg_tree.mapping.items():
        from_node, _ = lookup[key]
        for val in vals:
            to_node, graph = lookup[val]
            edge = cdg.CDGEdge(from_node, to_node, None, True)
            graph.add_edge(edge)

    return PDG(cur_cdg)
