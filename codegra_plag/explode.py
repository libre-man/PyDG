import ast
import copy as c
import typing as t

_gensym_cur = 0


def gensym():
    global _gensym_cur
    _gensym_cur += 1
    name = f'__G-{_gensym_cur}'
    return ast.Name(
        id=name, ctx=ast.Load()
    ), ast.Name(
        id=name, ctx=ast.Store()
    )


def maybe_store(
    expr: ast.expr, new_stmts: t.List[ast.stmt], force: bool=False
) -> ast.expr:
    res_load, res_store = gensym()
    new_stmts.append(ast.Assign(
        targets=[res_store],
        value=expr,
    ))
    return res_load


def is_simple(expr: ast.expr) -> bool:
    if isinstance(
        expr,
        (
            ast.Name, ast.Num, ast.Constant, ast.Str, ast.NameConstant,
            ast.Bytes, ast.Ellipsis
        )
    ):
        return True
    # elif isinstance(expr, ast.Attribute):
    #     return is_simple(expr.value) and is_simple(expr.attr)
    # elif isinstance(expr, ast.Index):
    #     return is_simple(expr.value)
    # elif isinstance(expr, ast.BoolOp):
    #     return all(is_simple(v) for v in expr.values)
    # elif isinstance(expr, ast.BinOp):
    #     return is_simple(expr.left) and is_simple(expr.right)
    # elif isinstance(expr, ast.UnaryOp):
    #     return is_simple(expr.operand)
    # elif isinstance(expr, (ast.Tuple, ast.List)):
    #     return all(is_simple(v) for v in expr.elts)
    else:
        return False


def explode_arguments(
    args: ast.arguments,
    new_stmts: t.List[ast.stmt],
) -> ast.arguments:
    args.defaults = [explode_expr(e, new_stmts) for e in args.defaults]
    args.kw_defaults = [explode_expr(e, new_stmts) for e in args.kw_defaults]
    return args


def explode_list_comp(
    comp: t.Union[ast.ListComp, ast.GeneratorExp, ast.SetComp],
    new_stmts: t.List[ast.stmt],
    init: ast.expr,
    appender_attr: str,
) -> ast.expr:
    res_load, res_store = gensym()
    res_stmts: t.List[ast.stmt] = [
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=c.deepcopy(res_load),
                    attr=appender_attr,
                    ctx=ast.Load()
                ),
                args=[comp.elt]
            )
        )
    ]

    for gen in reversed(comp.generators):
        new_loop: t.Union[ast.For, ast.AsyncFor]
        if gen.is_async:
            new_loop = ast.AsyncFor(target=gen.target)
        else:
            new_loop = ast.For(target=gen.target)

        body: t.List[ast.stmt] = []
        new_loop.iter = explode_expr(gen.iter, body)

        for cond in reversed(gen.ifs):
            res_stmts = [ast.If(cond, res_stmts)]

        new_loop.body = res_stmts

        body.append(new_loop)
        res_stmts = body

    new_stmts.append(ast.Assign(
        targets=[res_store],
        value=init,
    ))
    new_stmts.extend(res_stmts)
    return res_load


def explode_slice(slice: ast.slice, new_stmts: t.List[ast.stmt]) -> ast.slice:
    if isinstance(slice, ast.ExtSlice):
        slice.dims = [explode_slice(s, new_stmts) for s in slice.dims]
    elif isinstance(slice, ast.Slice):
        res = ':'
        if slice.lower:
            slice.lower = explode_expr(slice.lower, new_stmts)
        if slice.upper:
            slice.upper = explode_expr(slice.upper, new_stmts)
        if slice.step:
            slice.step = explode_expr(slice.step, new_stmts)
    elif isinstance(slice, ast.Index):
        slice.value = explode_expr(slice.value, new_stmts)
    else:
        assert False

    return slice


def explode_dict(expr: ast.Dict, new_stmts: t.List[ast.stmt]) -> ast.expr:
    if all(map(is_simple, expr.keys)) and all(map(is_simple, expr.values)):
        return expr

    res_load, res_store = gensym()
    new_dict = ast.Dict(keys=[], values=[])
    for key, value in zip(expr.keys, expr.values):
        key = explode_expr(key, new_stmts)
        value = explode_expr(value, new_stmts)
        new_dict.keys.append(key)
        new_dict.values.append(value)

    new_stmts.append(ast.Assign(targets=[res_store], value=new_dict))
    return res_load


def explode_expr(expr: ast.expr, new_stmts: t.List[ast.stmt]) -> ast.expr:
    if is_simple(expr):
        return expr

    if isinstance(expr, ast.BoolOp):
        expr.values = [explode_expr(v, new_stmts) for v in expr.values]
        return expr
    elif isinstance(expr, ast.BinOp):
        expr.left = explode_expr(expr.left, new_stmts)
        expr.right = explode_expr(expr.right, new_stmts)
        return maybe_store(expr, new_stmts)
    elif isinstance(expr, ast.UnaryOp):
        expr.operand = explode_expr(expr.operand, new_stmts)
        return maybe_store(expr, new_stmts)
    elif isinstance(expr, ast.Lambda):
        res_body: t.List[ast.stmt] = []
        res_expr = explode_expr(expr.body, res_body)
        res_body.append(ast.Return(value=res_expr))
        args = explode_arguments(expr.args, new_stmts)
        res_load, res_store = gensym()
        define = ast.FunctionDef(
            name=res_store.id,
            args=args,
            body=res_body,
            decorator_list=[],
            returns=None
        )
        new_stmts.append(define)
        return res_load
    elif isinstance(expr, ast.IfExp):
        res_load, res_store = gensym()

        if_body: t.List[ast.stmt] = []
        expr.body = explode_expr(expr.body, if_body)
        if_body.append(ast.Assign(targets=[res_store], value=expr.body))

        else_body: t.List[ast.stmt] = []
        expr.orelse = explode_expr(expr.orelse, else_body)
        else_body.append(
            ast.Assign(targets=[c.deepcopy(res_store)], value=expr.orelse)
        )

        new_stmts.append(
            ast.If(test=expr.test, body=if_body, orelse=else_body)
        )
        return res_load
    elif isinstance(expr, ast.Dict):
        return explode_dict(expr, new_stmts)
    elif isinstance(expr, ast.Set):
        expr.elts = [explode_expr(elt, new_stmts) for elt in expr.elts]
        return expr
    elif isinstance(expr, (ast.ListComp, ast.GeneratorExp)):
        return explode_list_comp(
            expr, new_stmts, ast.List(
                elts=[],
                ctx=ast.Load(),
            ), 'append'
        )
    elif isinstance(expr, ast.SetComp):
        return explode_list_comp(
            expr,
            new_stmts,
            ast.Set(
                elts=[],
                ctx=ast.Load(),
            ),
            'add',
        )
    elif isinstance(expr, ast.DictComp):
        return '{' + '{}: {} {}'.format(
            expr_to_str(expr.key),
            expr_to_str(expr.value),
            ' '.join(map(comprehension_to_str, expr.generators))
        ) + '}'
    elif isinstance(
        expr,
        ast.Starred,
    ):
        # This calls `__iter__` on `expr.value` which can have side effects. We
        # should prob. store this value.
        expr.value = explode_expr(expr.value, new_stmts)
        return expr
    elif isinstance(
        expr,
        (ast.Yield, ast.YieldFrom, ast.Await, ast.FormattedValue)
    ):
        expr.value = explode_expr(expr.value, new_stmts)
        return expr
    elif isinstance(expr, ast.Compare):
        expr.left = explode_expr(expr.left, new_stmts)
        expr.comparators = [
            explode_expr(comp, new_stmts) for comp in expr.comparators
        ]
        return maybe_store(expr, new_stmts)
    elif isinstance(expr, ast.Call):
        expr.func = explode_expr(expr.func, new_stmts)
        expr.args = [explode_expr(arg, new_stmts) for arg in expr.args]
        for keyword in expr.keywords:
            keyword.value = explode_expr(keyword.value, new_stmts)

        return maybe_store(expr, new_stmts)
    elif isinstance(
        expr,
        (
            ast.Num, ast.Bytes, ast.Str, ast.NameConstant, ast.Constant,
            ast.Ellipsis, ast.Name
        )
    ):
        assert False
    elif isinstance(expr, ast.JoinedStr):
        expr.values = [explode_expr(val, new_stmts) for val in expr.values]
        return expr
    elif isinstance(expr, ast.Attribute):
        expr.value = explode_expr(expr.value, new_stmts)
        return maybe_store(expr, new_stmts)
    elif isinstance(expr, ast.Subscript):
        expr.value = explode_expr(expr.value, new_stmts)
        expr.slice = explode_slice(expr.slice, new_stmts)
        return maybe_store(expr, new_stmts)
    elif isinstance(expr, (ast.List, ast.Tuple)):
        expr.elts = [explode_expr(elt, new_stmts) for elt in expr.elts]
        return expr
    else:
        assert False


def explode_with(stmt: ast.With) -> t.List[ast.stmt]:
    if stmt.items:
        res: t.List[ast.stmt] = []
        first: ast.withitem = stmt.items.pop(0)
        first.context_expr = explode_expr(first.context_expr, res)
        res.append(ast.With(items=[first], body=explode_with(stmt)))
        return res
    else:
        return explode_stmts(stmt.body)


def explode_target(expr: ast.expr, new_stmts: t.List[ast.stmt]) -> ast.expr:
    if isinstance(expr, ast.Name):
        return expr
    elif isinstance(expr, (ast.List, ast.Tuple)):
        expr.elts = [explode_target(elt, new_stmts) for elt in expr.elts]
        return expr
    elif isinstance(expr, (ast.Attribute, ast.Starred)):
        expr.value = explode_target(expr.value, new_stmts)
        return expr
    elif isinstance(expr, ast.Subscript):
        expr.value = explode_target(expr.value, new_stmts)
        expr.slice = explode_slice(expr.slice, new_stmts)
        return expr
    else:
        assert False


def explode_delete(stmt: ast.Delete) -> t.List[ast.stmt]:
    pass

def explode_stmts(stmts: t.List[ast.stmt]) -> t.List[ast.stmt]:
    res: t.List[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.FunctionDef): pass
        if isinstance(stmt, ast.AsyncFunctionDef): pass
        elif isinstance(stmt, ast.ClassDef): pass
        elif isinstance(stmt, ast.Return):
            if stmt.value:
                stmt.value = explode_expr(stmt.value, res)
        elif isinstance(stmt, ast.Delete):
            pass
        elif isinstance(stmt, ast.Assign): pass
        elif isinstance(stmt, ast.AugAssign): pass
        elif isinstance(stmt, ast.AnnAssign): pass
        elif isinstance(stmt, ast.For): pass
        elif isinstance(stmt, ast.AsyncFor): pass
        elif isinstance(stmt, (ast.If, ast.While)):
            stmt.test = explode_expr(stmt.test, res)
            stmt.body = explode_stmts(stmt.body)
            stmt.orelse = explode_stmts(stmt.orelse)
        elif isinstance(stmt, ast.With):
            res.extend(explode_with(stmt))
            continue
        elif isinstance(stmt, ast.AsyncWith): pass
        elif isinstance(stmt, ast.Raise):
            if stmt.exc:
                stmt.exc = explode_expr(stmt.exc, res)
            if stmt.cause:
                stmt.cause = explode_expr(stmt.cause, res)
        elif isinstance(stmt, ast.Try):
            stmt.body = explode_stmts(stmt.body)
            stmt.orelse = explode_stmts(stmt.orelse)
            stmt.finalbody = explode_stmts(stmt.finalbody)
            for handler in stmt.handlers:
                handler.body = explode_stmts(handler.body)
        elif isinstance(stmt, ast.Assert):
            stmt.test = explode_expr(stmt.test, res)
        elif isinstance(stmt, ast.Expr):
            stmt.value = explode_expr(stmt.value, res)
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
