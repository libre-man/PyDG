import ast
import copy as c
import typing as t
import logging
from copy import copy

_gensym_cur = 0


def gensym() -> t.Tuple[ast.Name, ast.Name]:
    global _gensym_cur
    _gensym_cur += 1
    name = f'__G-{_gensym_cur}'
    return ast.Name(
        id=name, ctx=ast.Load()
    ), ast.Name(
        id=name, ctx=ast.Store()
    )


def simplify_slice(slice: ast.slice, new_stmts: t.List[ast.stmt]) -> ast.slice:
    if isinstance(slice, ast.ExtSlice):
        slice.dims = [simplify_slice(s, new_stmts) for s in slice.dims]
    elif isinstance(slice, ast.Slice):
        res = ':'
        if slice.lower:
            slice.lower = SimplifyExpr(new_stmts).visit(slice.lower)
        if slice.upper:
            slice.upper = SimplifyExpr(new_stmts).visit(slice.upper)
        if slice.step:
            slice.step = SimplifyExpr(new_stmts).visit(slice.step)
    elif isinstance(slice, ast.Index):
        slice.value = SimplifyExpr(new_stmts).visit(slice.value)
    else:
        assert False

    return slice


class NameRewriter(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: t.Dict[str, str] = {}

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            return
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store):
            return
        else:
            self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.names:
            if isinstance(node.ctx, ast.Load):
                return
            self.names[node.id] = gensym()[0].id
        node.id = self.names[node.id]


def simplify_list_comp(
    comp: t.Union[ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp],
    new_stmts: t.List[ast.stmt],
    init: ast.expr,
    appender_attr: str,
) -> ast.Name:
    rewriter = NameRewriter()
    for gen in comp.generators:
        rewriter.visit(gen.target)

    res_load, res_store = gensym()
    res_stmts: t.List[ast.stmt] = []
    if isinstance(comp, ast.DictComp):
        rewriter.visit(comp.key)
        rewriter.visit(comp.value)
        key = SimplifyExpr(res_stmts).visit(comp.key)
        value = SimplifyExpr(res_stmts).visit(comp.value)
        res_stmts.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=c.deepcopy(res_load),
                        slice=ast.Index(value=key),
                        ctx=ast.Store()
                    )
                ],
                value=value
            )
        )
    else:
        rewriter.visit(comp.elt)
        comp.elt = SimplifyExpr(res_stmts).visit(comp.elt)
        res_stmts.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=c.deepcopy(res_load),
                        attr=appender_attr,
                        ctx=ast.Load()
                    ),
                    args=[comp.elt],
                    keywords=[],
                )
            )
        )

    for gen in reversed(comp.generators):
        new_loop: t.Union[ast.For, ast.AsyncFor]
        if hasattr(gen, 'is_async') and gen.is_async:  # type: ignore
            new_loop = ast.AsyncFor(target=gen.target, orelse=[])
        else:
            new_loop = ast.For(target=gen.target, orelse=[])

        body: t.List[ast.stmt] = []
        new_loop.iter = SimplifyExpr(body).visit(gen.iter)

        for cond in reversed(gen.ifs):
            res_stmts = [ast.If(test=cond, body=res_stmts, orelse=[])]

        new_loop.body = res_stmts

        body.append(new_loop)
        res_stmts = body

    new_stmts.append(ast.Assign(
        targets=[res_store],
        value=init,
    ))
    new_stmts.extend(res_stmts)
    return res_load


class SimplifyExpr(ast.NodeTransformer):
    def __init__(self, new_stmts: t.List[ast.stmt]) -> None:
        super().__init__()
        self.simplified = False
        self.new_stmts = new_stmts

    def reset(self) -> None:
        self.simplified = False

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.Name:
        self.simplified = True
        return simplify_list_comp(
            node, self.new_stmts, ast.List(
                elts=[],
                ctx=ast.Load(),
            ), 'append'
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.Name:
        self.simplified = True
        return simplify_list_comp(
            node, self.new_stmts, ast.List(
                elts=[],
                ctx=ast.Load(),
            ), 'append'
        )

    def visit_DictComp(self, node: ast.DictComp) -> ast.Name:
        self.simplified = True
        return simplify_list_comp(
            node,
            self.new_stmts,
            ast.Dict(
                keys=[],
                values=[],
            ),
            'UNUSED!',
        )

    def visit_SetComp(self, node: ast.SetComp) -> ast.Name:
        self.simplified = True
        return simplify_list_comp(
            node,
            self.new_stmts,
            ast.Set(
                elts=[],
            ),
            'add',
        )

    def visit_IfExp(self, node: ast.IfExp) -> ast.Name:
        self.simplified = True
        test = self.visit(node.test)
        left = self.visit(node.body)
        right = self.visit(node.orelse)
        res_load, res_store = gensym()

        self.new_stmts.append(
            ast.If(
                test=test,
                body=[ast.Assign(targets=[res_store], value=left)],
                orelse=[ast.Assign(targets=[copy(res_store)], value=right)]
            )
        )
        return res_load


def simplify_with(stmt: ast.With) -> t.List[ast.stmt]:
    if stmt.items:
        res: t.List[ast.stmt] = []
        first: ast.withitem = stmt.items.pop(0)
        first.context_expr = SimplifyExpr(res).visit(first.context_expr)
        res.append(ast.With(items=[first], body=simplify_with(stmt)))
        return res
    else:
        return simplify_stmts(stmt.body)


def simplify_target(expr: ast.expr, new_stmts: t.List[ast.stmt]) -> ast.expr:
    if isinstance(expr, ast.Name):
        return expr
    elif isinstance(expr, (ast.List, ast.Tuple)):
        expr.elts = [simplify_target(elt, new_stmts) for elt in expr.elts]
        return expr
    elif isinstance(expr, (ast.Attribute, ast.Starred)):
        expr.value = simplify_target(expr.value, new_stmts)
        return expr
    elif isinstance(expr, ast.Subscript):
        expr.value = simplify_target(expr.value, new_stmts)
        expr.slice = simplify_slice(expr.slice, new_stmts)
        return expr
    else:
        assert False


def simplify_stmts(stmts: t.List[ast.stmt]) -> t.List[ast.stmt]:
    res: t.List[ast.stmt] = []
    simplifier = SimplifyExpr(res)

    for stmt in stmts:
        if isinstance(
            stmt,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            stmt.body = simplify_stmts(stmt.body)
        elif isinstance(stmt, ast.Module):
            stmt.body = simplify_stmts(stmt.body)
        elif isinstance(stmt, ast.Return):
            if stmt.value:
                stmt.value = simplifier.visit(stmt.value)
        elif isinstance(stmt, ast.Delete):
            pass
        elif isinstance(
            stmt,
            (
                ast.AugAssign,
                ast.AnnAssign,  # type: ignore
                ast.Assign,
            )
        ):
            if stmt.value:
                stmt.value = simplifier.visit(stmt.value)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            stmt.iter = simplifier.visit(stmt.iter)
            stmt.body = simplify_stmts(stmt.body)
            stmt.orelse = simplify_stmts(stmt.orelse)
        elif isinstance(stmt, ast.While):
            stmt.body = simplify_stmts(stmt.body)
            stmt.orelse = simplify_stmts(stmt.orelse)

            pre_test: t.List[ast.stmt] = []
            s = SimplifyExpr(pre_test)
            test = s.visit(stmt.test)
            if s.simplified:
                if stmt.orelse:
                    logging.warn(
                        'While with complicated test and else is not supported'
                    )
                    pass
                stmt.body = pre_test + [ast.If(test=test, body=[ast.Break()])
                                        ] + stmt.body
                stmt.test = ast.NameConstant(value=True)
        elif isinstance(stmt, ast.If):
            stmt.test = simplifier.visit(stmt.test)
            stmt.body = simplify_stmts(stmt.body)
            stmt.orelse = simplify_stmts(stmt.orelse)
        elif isinstance(stmt, ast.With):
            res.extend(simplify_with(stmt))
            continue
        elif isinstance(stmt, ast.AsyncWith):
            pass
        elif isinstance(stmt, ast.Raise):
            if stmt.exc:
                stmt.exc = simplifier.visit(stmt.exc)
            if stmt.cause:
                stmt.cause = simplifier.visit(stmt.cause)
        elif isinstance(stmt, ast.Try):
            stmt.body = simplify_stmts(stmt.body)
            stmt.orelse = simplify_stmts(stmt.orelse)
            stmt.finalbody = simplify_stmts(stmt.finalbody)
            for handler in stmt.handlers:
                handler.body = simplify_stmts(handler.body)
        elif isinstance(stmt, ast.Assert):
            stmt.test = simplifier.visit(stmt.test)
        elif isinstance(stmt, ast.Expr):
            stmt.value = simplifier.visit(stmt.value)
        elif isinstance(
            stmt,
            (
                ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal,
                ast.Import, ast.ImportFrom
            )
        ):
            pass
        else:
            assert False

        res.append(stmt)
    return res
