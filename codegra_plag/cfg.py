"""This module creates a CFG for an module. Use the ``create_cfg`` to do this,
all other functions are helper functions.
"""
import ast
import typing as t

import astor  # type: ignore

import graph


class CFGNode(graph.Node):
    pass


class CFG(graph.Graph[CFGNode]):
    def simplify(self) -> None:
        pass

CFGEdge = graph.Edge[CFGNode]


class CFGPartialEdge:
    def __init__(self, head: CFGNode, label: t.Optional[str], graph: CFG) -> None:
        self.head = head
        self.label = label
        self.graph = graph

    def add_tail(self, tail: CFGNode, label: t.Optional[str]=None) -> CFGEdge:
        if label:
            label = '{} - {}'.format(
                self.label, label
            ) if self.label else label
        else:
            label = self.label
        edge = graph.Edge(self.head, tail, label)
        self.graph.add_edge(edge)
        return edge


T = t.TypeVar('T')


def process_block(
    stmts: t.Iterable[ast.stmt],
    start: t.List[CFGPartialEdge],
    loop_start: t.Optional[CFGNode],
    loop_end: t.List[CFGPartialEdge],
    fun_end: CFGNode,
    res_graph: CFG,
) -> t.List[CFGPartialEdge]:
    """Process a block of statements.
    """
    prev_tails = start

    for stmt in stmts:
        head, tails = create_cfg_stmt(
            stmt, loop_start, loop_end, fun_end, res_graph
        )
        if head is None:
            continue

        for prev_tail in prev_tails:
            prev_tail.add_tail(head)

        prev_tails = tails
        if not tails:
            break

    return prev_tails


HeadTails = t.Tuple[t.Optional[CFGNode], t.List[CFGPartialEdge]]


def create_cfg_try(
    stmt: ast.Try,
    loop_start: t.Optional[CFGNode],
    loop_end: t.List[CFGPartialEdge],
    fun_end: CFGNode,
    res_graph: CFG,
) -> HeadTails:
    """Create a CFG for a try block.
    """
    head_node = CFGNode('Try')
    try_tails = process_block(
        stmt.body,
        [CFGPartialEdge(head_node, None, res_graph)],
        loop_start,
        loop_end,
        fun_end,
        res_graph,
    )
    tails: t.List[CFGPartialEdge] = []

    for handler in stmt.handlers:
        if handler.type and handler.name:
            node = CFGNode(
                'Except {} as {}'.format(
                    astor.to_source(handler.type).strip(),
                    astor.to_source(handler.name).strip()
                )
            )
        elif handler.type:
            node = CFGNode(
                'Except {}'.format(astor.to_source(handler.type).strip())
            )
        else:
            node = CFGNode('Except')
        for tail in try_tails:
            tail.add_tail(node, 'Exception')
        tails += process_block(
            handler.body,
            [CFGPartialEdge(node, None, res_graph)],
            loop_start,
            loop_end,
            fun_end,
            res_graph,
        )

    if stmt.orelse:
        node = CFGNode('Else')
        for tail in try_tails:
            tail.add_tail(node, 'No Exception')
        tails += process_block(
            stmt.orelse,
            [CFGPartialEdge(node, None, res_graph)],
            loop_start,
            loop_end,
            fun_end,
            res_graph,
        )
    else:
        for tt in try_tails:
            tt.label = 'No Exception'
        tails += try_tails

    if stmt.finalbody:
        node = CFGNode('Finally')
        for tail in tails:
            tail.add_tail(node, 'Finally')
        tails = process_block(
            stmt.finalbody,
            [CFGPartialEdge(node, None, res_graph)],
            loop_start,
            loop_end,
            fun_end,
            res_graph,
        )

    return head_node, tails


def create_cfg_class(
    cls: t.Union[ast.Module, ast.ClassDef],
    res_graph: CFG,
) -> None:
    top_fun_stmts: t.List[ast.stmt] = []

    for stmt in cls.body:
        if isinstance(stmt, ast.ClassDef):
            create_cfg_class(stmt, res_graph)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            create_cfg_stmt(stmt, None, [], None, res_graph)
        else:
            top_fun_stmts.append(stmt)

    create_cfg_stmt(
        ast.FunctionDef(
            name="__at_read_time__",
            args=ast.arguments(
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=top_fun_stmts,
            decorator_list=[],
            returns=None
        ), None, [], None, res_graph
    )


def create_cfg_loop(
    loop: t.Union[ast.While, ast.For, ast.AsyncFor],
    loop_start: t.Optional[CFGNode],
    loop_end: t.List[CFGPartialEdge],
    fun_end: CFGNode,
    res_graph: CFG,
) -> HeadTails:
    if isinstance(loop, ast.While):
        labels = 'True', 'Else'
        node = CFGNode('While {}'.format(astor.to_source(loop.test).strip()))
    else:
        labels = 'Cont', 'End'
        node = CFGNode(
            'For {} in {}'.format(
                astor.to_source(loop.target).strip(),
                astor.to_source(loop.iter).strip()
            )
        )

    tails: t.List[CFGPartialEdge] = []
    for tail in process_block(
        loop.body, [CFGPartialEdge(node, labels[0], res_graph)], node, tails, fun_end,
        res_graph
    ):
        tail.add_tail(node)

    if loop.orelse:
        extra_tails = process_block(
            loop.orelse, [CFGPartialEdge(node, labels[1], res_graph)], loop_start,
            loop_end, fun_end, res_graph
        )
    else:
        extra_tails = [CFGPartialEdge(node, labels[1], res_graph)]

    return node, tails + extra_tails


def create_cfg_stmt(
    stmt: ast.stmt,
    loop_start: t.Optional[CFGNode],
    loop_end: t.List[CFGPartialEdge],
    fun_end: t.Optional[CFGNode],
    res_graph: CFG,
) -> HeadTails:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        g = CFG()
        start = CFGNode('FIRST ({})'.format(stmt.name))
        end = CFGNode('END')
        for tail in process_block(
            stmt.body,
            [CFGPartialEdge(start, None, g)],
            None,
            [],
            end,
            g,
        ):
            tail.add_tail(end)

        res_graph.add_subgraph(g)
        return None, []
    else:
        assert isinstance(fun_end, CFGNode)

    if isinstance(stmt, ast.ClassDef):
        create_cfg_class(stmt, res_graph)
        return None, []
    elif isinstance(stmt, ast.Continue):
        node = CFGNode('Continue')
        assert loop_start
        res_graph.add_edge(CFGEdge(node, loop_start, None))
        return node, []
    elif isinstance(stmt, ast.Break):
        node = CFGNode('Break')
        assert loop_end is not None
        loop_end.append(CFGPartialEdge(node, None, res_graph))
        return node, []
    elif isinstance(stmt, ast.Return):
        node = CFGNode(astor.to_source(stmt).strip())
        assert fun_end
        res_graph.add_edge(CFGEdge(node, fun_end, None))
        return node, []
    elif isinstance(stmt, ast.Raise):
        node = CFGNode(astor.to_source(stmt).strip())
        assert fun_end
        res_graph.add_edge(CFGEdge(node, fun_end, None))
        return node, []
    elif isinstance(stmt, ast.Assert):
        node = CFGNode(astor.to_source(stmt).strip())
        assert fun_end
        res_graph.add_edge(CFGEdge(node, fun_end, 'False'))
        return node, [CFGPartialEdge(node, 'True', res_graph)]
    elif isinstance(
        stmt,
        (
            ast.Delete,
            ast.Assign,
            ast.Import,
            ast.ImportFrom,
            ast.Nonlocal,
            ast.Global,
            ast.Expr,
            ast.AugAssign,
            ast.Pass,
            ast.AnnAssign  # type: ignore
        )
    ):
        node = CFGNode(astor.to_source(stmt).strip())
        return node, [CFGPartialEdge(node, None, res_graph)]
    elif isinstance(stmt, ast.If):
        node = CFGNode('If {}'.format(astor.to_source(stmt.test).strip()))
        then_tails = process_block(
            stmt.body,
            [CFGPartialEdge(node, 'True', res_graph)],
            loop_start,
            loop_end,
            fun_end,
            res_graph,
        )
        if stmt.orelse:
            else_tails = process_block(
                stmt.orelse, [CFGPartialEdge(node, 'Else', res_graph)], loop_start,
                loop_end, fun_end, res_graph
            )
        else:
            else_tails = [CFGPartialEdge(node, 'Else', res_graph)]

        return (node, then_tails + else_tails)
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return create_cfg_loop(stmt, loop_start, loop_end, fun_end, res_graph)
    elif isinstance(stmt, (ast.AsyncWith, ast.With)):
        node = CFGNode(
            'With {}'.format(
                ', '.
                join(astor.to_source(item).strip() for item in stmt.items)
            )
        )
        return node, process_block(
            stmt.body,
            [CFGPartialEdge(node, None, res_graph)],
            loop_start,
            loop_end,
            fun_end,
            res_graph,
        )
    elif isinstance(stmt, ast.Try):
        return create_cfg_try(stmt, loop_start, loop_end, fun_end, res_graph)
    else:
        print(stmt)
        assert False


def create_cfg(tree: ast.Module) -> CFG:
    """Create a CFG for an entire module.
    """
    res_graph = CFG()
    create_cfg_class(tree, res_graph)
    return res_graph
