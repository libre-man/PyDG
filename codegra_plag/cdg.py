import ast
import enum
import typing as t
from contextlib import contextmanager

import astor  # type: ignore

from . import graph


class CDGNode(graph.Node):
    def __init__(self, orig: t.Union[ast.stmt, ast.ExceptHandler],
                 label: str) -> None:
        super().__init__(label)
        self.orig = orig


class CDG(graph.Graph[CDGNode]):
    def flatten(self) -> None:
        todo = list(self.subgraphs)
        while todo:
            cur = todo.pop()
            for s in cur.subgraphs:
                todo.append(s)
                self.subgraphs.append(s)
            cur.subgraphs = []

    def get_all_tails_from_node(self, from_node: CDGNode,
                                dotted_only: bool) -> t.Iterable[int]:
        for edge in self.edges:
            if dotted_only and not edge.dotted:
                continue
            if edge.head == from_node:
                i = 0
                for node in self.get_not_special_nodes():
                    if node == edge.tail:
                        yield i
                        break
                    i += 1

    def get_not_special_nodes(self) -> t.Iterable[CDGNode]:
        for node in self.nodes:
            if isinstance(node, CDGRegionNode) and node.type not in {
                CDGRegionNodeType.enter, CDGRegionNodeType.loop_block,
                CDGRegionNodeType.if_block, CDGRegionNodeType.except_block
            }:
                continue
            yield node

    def to_mcgregor(self) -> t.List[str]:
        mapping: t.Dict[CDGNode, int] = {}
        out: t.List[str] = []
        for idx, node in enumerate(self.nodes):
            mapping[node] = idx
            out.append(str(idx))
        for edge in self.edges:
            out.append('{},{}'.format(mapping[edge.head], mapping[edge.tail]))
        return out


CDGEdge = graph.Edge[CDGNode]


class CDGRegionNodeType(enum.Enum):
    try_block = enum.auto()
    try_body = enum.auto()
    except_block = enum.auto()
    finally_block = enum.auto()
    else_block = enum.auto()
    if_block = enum.auto()
    then_block = enum.auto()
    with_block = enum.auto()
    loop_block = enum.auto()
    loop_body = enum.auto()
    enter = enum.auto()
    exit = enum.auto()


class CDGRegionNode(CDGNode):
    def __init__(
        self,
        orig: t.Union[ast.stmt, ast.ExceptHandler],
        type: CDGRegionNodeType,
        name: t.Optional[str]=None,
        extra: t.Optional[str]=None,
        stmt: t.Optional[ast.stmt]=None,
    ) -> None:
        super().__init__(orig, name or str(type))
        self.breaks: t.List[CDGNode] = []
        self.type = type
        self.extra = extra
        self.stmt = stmt


class CDGStack(t.List[CDGRegionNode]):
    def __init__(self, graph: CDG) -> None:
        super().__init__()
        self.graph = graph
        self.connect_to_next: t.List[CDGNode] = []

    def add_node(self, node: CDGNode, label: t.Optional[str]=None) -> None:
        self.graph.add_edge(CDGEdge(self[-1], node, label))

    def get_loop(self, amount: int) -> CDGRegionNode:
        for node in reversed(self):
            if node.type == CDGRegionNodeType.loop_block:
                if amount == 1:
                    return node
                else:
                    amount -= 1
        raise ValueError

    @contextmanager
    def pushed(self, item: CDGRegionNode,
               label: t.Optional[str]=None) -> t.Iterator[CDGRegionNode]:
        if len(self) == 0:
            assert item.type == CDGRegionNodeType.enter
        else:
            self.graph.add_edge(graph.Edge(self[-1], item, label))
        self.append(item)
        yield item
        self.pop()


def process_block(
    stmts: t.Iterable[ast.stmt],
    cdg_stack: CDGStack,
    exit_node: CDGRegionNode,
    res_graph: CDG,
) -> None:
    do_connect: t.List[CDGNode] = []
    for stmt in stmts:
        new = create_cdg_stmt(
            stmt,
            cdg_stack,
            exit_node,
            res_graph,
        )
        if do_connect and new is not None:
            for node in do_connect:
                res_graph.add_edge(CDGEdge(node, new, None))
            do_connect = []
        if cdg_stack.connect_to_next:
            do_connect.extend(cdg_stack.connect_to_next)
            cdg_stack.connect_to_next = []
        if isinstance(stmt, ast.Return):
            break

    if do_connect:
        cdg_stack.connect_to_next.extend(do_connect)


def create_cdg_try(
    stmt: ast.Try,
    cdg_stack: CDGStack,
    exit_node: CDGRegionNode,
    res_graph: CDG,
) -> CDGNode:
    top_node = CDGRegionNode(ast.Pass(), CDGRegionNodeType.try_block)
    with cdg_stack.pushed(top_node):
        with cdg_stack.pushed(
            CDGRegionNode(ast.Pass(), CDGRegionNodeType.try_body)
        ):
            process_block(stmt.body, cdg_stack, exit_node, res_graph)

        to_else = cdg_stack.connect_to_next
        to_finally = []
        cdg_stack.connect_to_next = []

        for handler in stmt.handlers:
            if handler.type and handler.name:
                t = astor.to_source(handler.type).strip()
                label = 'Except {} as {}'.format(
                    astor.to_source(handler.type).strip(), handler.name.strip()
                )
            elif handler.type:
                t = astor.to_source(handler.type).strip()
                label = 'Except {}'.format(
                    astor.to_source(handler.type).strip()
                )
            else:
                t = 'Any'
                label = 'Except'

            node = CDGRegionNode(
                handler, CDGRegionNodeType.except_block, label
            )
            with cdg_stack.pushed(node, 'Except {}'.format(t)):
                process_block(handler.body, cdg_stack, exit_node, res_graph)
            to_finally.extend(cdg_stack.connect_to_next)
            cdg_stack.connect_to_next = []

        if stmt.orelse:
            node = CDGRegionNode(ast.Pass(), CDGRegionNodeType.else_block)
            for else_node in to_else:
                res_graph.add_edge(CDGEdge(else_node, node, None))
            with cdg_stack.pushed(node, 'No exception'):
                process_block(stmt.orelse, cdg_stack, exit_node, res_graph)
            to_finally.extend(cdg_stack.connect_to_next)
            cdg_stack.connect_to_next = []
        else:
            to_finally.extend(to_else)

        if stmt.finalbody:
            node = CDGRegionNode(ast.Pass(), CDGRegionNodeType.finally_block)
            for f_node in to_finally:
                res_graph.add_edge(CDGEdge(f_node, node, None))
            with cdg_stack.pushed(node, 'Finally'):
                process_block(stmt.finalbody, cdg_stack, exit_node, res_graph)
    return top_node


def create_cdg_class(
    cls: t.Union[ast.Module, ast.ClassDef],
    res_graph: CDG,
) -> None:
    top_fun_stmts: t.List[ast.stmt] = []

    stack = CDGStack(res_graph)

    for stmt in cls.body:
        if isinstance(stmt, ast.ClassDef):
            create_cdg_class(stmt, res_graph)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            create_cdg_stmt(stmt, stack, None, res_graph)
        else:
            top_fun_stmts.append(stmt)
        assert not stack

    if top_fun_stmts:
        create_cdg_stmt(
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
            ), stack, None, res_graph
        )


def create_cdg_loop(
    loop: t.Union[ast.While, ast.For, ast.AsyncFor],
    stack: CDGStack,
    exit_node: CDGRegionNode,
    res_graph: CDG,
) -> CDGNode:
    if isinstance(loop, ast.While):
        label = 'T'
        name = 'While {}'.format(astor.to_source(loop.test).strip())
    else:
        label = 'Cont'
        name = 'For {} in {}'.format(
            astor.to_source(loop.target).strip(),
            astor.to_source(loop.iter).strip()
        )

    loop_block = CDGRegionNode(loop, CDGRegionNodeType.loop_block, name)
    with stack.pushed(loop_block):
        loop_body = CDGRegionNode(ast.Pass(), CDGRegionNodeType.loop_body)
        res_graph.add_edge(graph.Edge(loop_body, loop_block, None))
        with stack.pushed(loop_body, label):
            process_block(loop.body, stack, exit_node, res_graph)

    if loop.orelse:
        with stack.pushed(
            CDGRegionNode(ast.Pass(), CDGRegionNodeType.else_block)
        ):
            process_block(loop.orelse, stack, exit_node, res_graph)

    for breaked in loop_block.breaks:
        stack.connect_to_next.append(breaked)
    loop_block.breaks = []
    return loop_block


def create_cdg_stmt(
    stmt: ast.stmt,
    stack: CDGStack,
    exit_node: t.Optional[CDGRegionNode],
    res_graph: CDG,
) -> t.Optional[CDGNode]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        g = CDG()
        stack = CDGStack(g)
        enter = CDGRegionNode(
            stmt,
            CDGRegionNodeType.enter,
            'ENTER def {}({})'.format(
                stmt.name, astor.to_source(stmt.args).strip()
            ),
            extra=astor.to_source(stmt),
            stmt=stmt
        )
        exit_node = CDGRegionNode(ast.Pass(), CDGRegionNodeType.exit)
        with stack.pushed(enter):
            process_block(
                stmt.body,
                stack,
                exit_node,
                g,
            )

        g.add_edge(CDGEdge(enter, exit_node, None))
        for node in stack.connect_to_next:
            g.add_edge(CDGEdge(node, exit_node, None))
        res_graph.add_subgraph(g)
        return None
    else:
        assert isinstance(exit_node, CDGRegionNode)

    if isinstance(stmt, ast.ClassDef):
        create_cdg_class(stmt, res_graph)
        return None
    elif isinstance(stmt, ast.Continue):
        node = CDGNode(stmt, 'Continue')
        res_graph.add_edge(CDGEdge(node, stack.get_loop(1), None))
        stack.add_node(node)
        return node
    elif isinstance(stmt, ast.Break):
        node = CDGNode(stmt, 'Break')

        try:
            loop = stack.get_loop(2)
            res_graph.add_edge(CDGEdge(node, loop, None))
        except ValueError:
            loop = stack.get_loop(1)
            loop.breaks.append(node)

        stack.add_node(node)
        return node
    elif isinstance(stmt, ast.Return):
        node = CDGNode(stmt, astor.to_source(stmt).strip())
        stack.add_node(node)
        res_graph.add_edge(CDGEdge(node, exit_node, None))
        return node
    elif isinstance(stmt, ast.Raise):
        node = CDGNode(stmt, astor.to_source(stmt).strip())
        stack.add_node(node)
        res_graph.add_edge(CDGEdge(node, exit_node, None))
        return node
    elif isinstance(stmt, ast.Assert):
        node = CDGNode(stmt, astor.to_source(stmt).strip())
        stack.add_node(node)
        res_graph.add_edge(CDGEdge(node, exit_node, 'F'))
        return node
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
        node = CDGNode(stmt, astor.to_source(stmt).strip())
        stack.add_node(node)
        return node
    elif isinstance(stmt, ast.If):
        node = CDGRegionNode(
            stmt, CDGRegionNodeType.if_block,
            'If {}'.format(astor.to_source(stmt.test).strip())
        )
        with stack.pushed(node):
            with stack.pushed(
                CDGRegionNode(ast.Pass(), CDGRegionNodeType.then_block), 'T'
            ):
                process_block(stmt.body, stack, exit_node, res_graph)
            if stmt.orelse:
                with stack.pushed(
                    CDGRegionNode(ast.Pass(), CDGRegionNodeType.else_block),
                    'F'
                ):
                    process_block(stmt.orelse, stack, exit_node, res_graph)
        return node
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return create_cdg_loop(stmt, stack, exit_node, res_graph)
    elif isinstance(stmt, (ast.AsyncWith, ast.With)):
        node = CDGNode(
            stmt, 'With {}'.format(
                ', '.
                join(astor.to_source(item).strip() for item in stmt.items)
            )
        )
        with stack.pushed(CDGRegionNode(stmt, CDGRegionNodeType.with_block)):
            process_block(
                stmt.body,
                stack,
                exit_node,
                res_graph,
            )
        return node
    elif isinstance(stmt, ast.Try):
        return create_cdg_try(stmt, stack, exit_node, res_graph)
    else:
        print(stmt)
        assert False


def create_cdg(tree: ast.Module) -> CDG:
    res_graph = CDG()
    create_cdg_class(tree, res_graph)
    res_graph.flatten()
    return res_graph
