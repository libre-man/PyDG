import ast
import typing as t


def boolop_to_str(boolop: ast.boolop) -> str:
    if isinstance(boolop, ast.And): return 'and'
    elif isinstance(boolop, ast.Or): return 'or'
    else: assert False


def unaryop_to_str(unaryop: ast.unaryop) -> str:
    if isinstance(unaryop, ast.Invert): return '~'
    elif isinstance(unaryop, ast.Not): return 'not'
    elif isinstance(unaryop, ast.UAdd): return '+'
    elif isinstance(unaryop, ast.USub): return '-'
    else: assert False


def binop_to_str(binop: ast.operator) -> str:
    if isinstance(binop, ast.Add): return '+'
    elif isinstance(binop, ast.Sub): return '-'
    elif isinstance(binop, ast.Mult): return '*'
    elif isinstance(binop, ast.MatMult): return '@'
    elif isinstance(binop, ast.Div): return '/'
    elif isinstance(binop, ast.Mod): return '%'
    elif isinstance(binop, ast.Pow): return '**'
    elif isinstance(binop, ast.LShift): return '<<'
    elif isinstance(binop, ast.RShift): return '>>'
    elif isinstance(binop, ast.BitOr): return '|'
    elif isinstance(binop, ast.BitXor): return '^'
    elif isinstance(binop, ast.BitAnd): return '&'
    elif isinstance(binop, ast.FloorDiv): return '//'
    else: assert False


def cmpop_to_str(cmpop: ast.cmpop) -> str:
    if isinstance(cmpop, ast.Eq): return '=='
    elif isinstance(cmpop, ast.NotEq): return '!='
    elif isinstance(cmpop, ast.Lt): return '<'
    elif isinstance(cmpop, ast.LtE): return '<='
    elif isinstance(cmpop, ast.Gt): return '>'
    elif isinstance(cmpop, ast.GtE): return '>='
    elif isinstance(cmpop, ast.Is): return 'is'
    elif isinstance(cmpop, ast.IsNot): return 'is not'
    elif isinstance(cmpop, ast.In): return 'in'
    elif isinstance(cmpop, ast.NotIn): return 'not in'
    else: assert False


def arguments_to_str(arguments: ast.arguments) -> str:
    res: t.List[str] = []
    for arg in arguments.args:
        res.append(arg.arg)
    if arguments.vararg:
        res.append('*{}'.format(arguments.vararg.arg))
    for arg in arguments.kwonlyargs:
        res.append(arg.arg)
    if arguments.kwarg:
        res.append('**{}'.format(arguments.kwarg.arg))

    return ', '.join(res)


def comprehension_to_str(comp: ast.comprehension) -> str:
    res = 'for {} in {}'.format(
        expr_to_str(comp.target), expr_to_str(comp.iter)
    )
    if comp.ifs:
        res += 'if ' + ' if '.join(map(expr_to_str, comp.ifs))
    return res


def slice_to_str(slice: ast.slice) -> str:
    if isinstance(slice, ast.ExtSlice):
        return ', '.join(map(slice_to_str, slice.dims))
    elif isinstance(slice, ast.Slice):
        res = ':'
        if slice.lower:
            res = expr_to_str(slice.lower) + res
        if slice.upper:
            res += expr_to_str(slice.upper)
        if slice.step:
            res += ':{}'.format(expr_to_str(slice.step))
        return res
    elif isinstance(slice, ast.Index):
        return expr_to_str(slice.value)
    else:
        assert False


def keyword_to_str(keyword: ast.keyword) -> str:
    if keyword.arg is None:
        return '**{}'.format(expr_to_str(keyword.value))
    else:
        return '{}={}'.format(str(keyword.arg), expr_to_str(keyword.value))

def expr_to_str(expr: ast.expr) -> str:
    if isinstance(expr, ast.BoolOp):
        return ' {} '.format(
            boolop_to_str(expr.op),
        ).join(map(expr_to_str, expr.values))
    elif isinstance(expr, ast.BinOp):
        return '({} {} {})'.format(
            expr_to_str(expr.left),
            binop_to_str(expr.op),
            expr_to_str(expr.right),
        )
    elif isinstance(expr, ast.UnaryOp):
        return '({} {})'.format(
            unaryop_to_str(expr.op),
            expr_to_str(expr.operand),
        )
    elif isinstance(expr, ast.Lambda):
        return 'lambda {}: {}'.format(
            arguments_to_str(expr.args),
            expr_to_str(expr.body),
        )
    elif isinstance(expr, ast.IfExp):
        return '({} if {} else {})'.format(
            expr_to_str(expr.body),
            expr_to_str(expr.test),
            expr_to_str(expr.orelse),
        )
    elif isinstance(expr, ast.Dict):
        body = (
            '{}: {}'.format(*val)
            for val in
            zip(map(expr_to_str, expr.keys), map(expr_to_str, expr.values))
        )
        return '{' + ', '.join(body) + '}'
    elif isinstance(expr, ast.Set):
        return '{' + ', '.join(map(expr_to_str, expr.elts)) + '}'
    elif isinstance(expr, ast.ListComp):
        return '[{} {}]'.format(
            expr_to_str(expr.elt),
            ' '.join(map(comprehension_to_str, expr.generators))
        )
    elif isinstance(expr, ast.SetComp):
        return '{' + '{} {}'.format(
            expr_to_str(expr.elt),
            ' '.join(map(comprehension_to_str, expr.generators))
        ) + '}'
    elif isinstance(expr, ast.DictComp):
        return '{' + '{}: {} {}'.format(
            expr_to_str(expr.key),
            expr_to_str(expr.value),
            ' '.join(map(comprehension_to_str, expr.generators))
        ) + '}'
    elif isinstance(expr, ast.GeneratorExp):
        return '({} {})'.format(
            expr_to_str(expr.elt),
            ' '.join(map(comprehension_to_str, expr.generators))
        )
    elif isinstance(expr, ast.Await):
        return 'await {}'.format(expr_to_str(expr.value))
    elif isinstance(expr, ast.Yield):
        if expr.value is None:
            return 'yield'
        else:
            return 'yield {}'.format(expr_to_str(expr.value))
    elif isinstance(expr, ast.YieldFrom):
        return 'yield from {}'.format(expr_to_str(expr.value))
    elif isinstance(expr, ast.Compare):
        return '({}{})'.format(
            expr_to_str(expr.left), ''.join(
                ' {} {}'.format(cmpop_to_str(op), expr_to_str(val))
                for op, val in zip(expr.ops, expr.comparators)
            )
        )
    elif isinstance(expr, ast.Call):
        return '{}({}, {})'.format(
            expr_to_str(expr.func),
            ', '.join(map(expr_to_str, expr.args)),
            ', '.join(map(keyword_to_str, expr.keywords)),
        )
    elif isinstance(expr, ast.Num):
        return str(expr.n)
    elif isinstance(expr, ast.Str):
        return '"{}"'.format(expr.s)
    elif isinstance(expr, ast.FormattedValue):
        return expr_to_str(expr.value)
    elif isinstance(expr, ast.JoinedStr):
        return ' + '.join(map(expr_to_str, expr.values))
    elif isinstance(expr, ast.Bytes):
        return str(expr.s)
    elif isinstance(expr, ast.NameConstant):
        return str(expr.value)
    elif isinstance(expr, ast.Ellipsis):
        return '...'
    elif isinstance(expr, ast.Constant):
        return str(expr.constant)
    elif isinstance(expr, ast.Attribute):
        return '{}.{}'.format(expr_to_str(expr.value), expr.attr)
    elif isinstance(expr, ast.Subscript):
        return '{}[{}]'.format(
            expr_to_str(expr.value), slice_to_str(expr.slice)
        )
    elif isinstance(expr, ast.Starred):
        return '*{}'.format(expr_to_str(expr.value))
    elif isinstance(expr, ast.Name):
        return expr.id
    elif isinstance(expr, ast.List):
        return '[{}]'.format(', '.join(map(expr_to_str, expr.elts)))
    elif isinstance(expr, ast.Tuple):
        return '({})'.format(', '.join(map(expr_to_str, expr.elts)))
    else:
        assert False
