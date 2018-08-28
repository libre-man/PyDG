"""This module creates a DDG for an module. Use the ``create_ddg`` to do this,
all other functions are helper functions.
"""
import ast
import collections
import logging
import typing as t
from contextlib import contextmanager
from copy import copy
from itertools import chain

T = t.TypeVar('T')
Y = t.TypeVar('Y', bound='DefMap')


def noop(*args: t.Any, **kwargs: t.Any) -> None:
    pass

@contextmanager
def no_warn() -> t.Iterator[None]:
    """
    Execute some block of code without logging warnings.
    """
    if logging.warn is noop:
        yield
    else:
        old = logging.warn
        logging.warn = noop
        yield
        logging.warn = old


# These options are not completely implemented yet, so simply set them all to
# ``True``.
Options = t.NamedTuple(
    'Options',
    [
        ('access_props', bool),
        ('set_props', bool),
        ('mutate_recursive', bool),
        ('method_args', bool),
        ('pure_operators', bool),
    ],
)

Mutation = t.Union[ast.stmt, ast.ExceptHandler]
Mutations = t.Set[Mutation]
FMutations = t.FrozenSet[Mutation]


class DefMap:
    __slots__ = ['_lookup', '_aliases']

    class Value:
        __slots__ = ['mutations', 'submap']

        def __init__(self, mutations: Mutations, submap: 'DefMap') -> None:
            self.mutations = mutations
            self.submap = submap

        def get_all_mutations(
            self,
            recurse: bool,
        ) -> t.Iterable[t.Union[ast.stmt, ast.ExceptHandler]]:
            yield from self.mutations
            if recurse:
                yield from flatten(
                    v.get_all_mutations(True)
                    for v in self.submap._lookup.values()
                )

        def merge_into(self, other: 'DefMap.Value') -> None:
            self.mutations.update(copy(other.mutations))
            self.submap = DefMap.merge(self.submap, other.submap)

        def copy(self) -> 'DefMap.Value':
            return DefMap.Value(copy(self.mutations), self.submap.copy())

    def __init__(self) -> None:
        self._lookup: t.DefaultDict[
            str, DefMap.Value
        ] = collections.defaultdict(lambda: DefMap.Value(set(), DefMap()))
        self._aliases: t.Dict['Var', t.Set[t.Tuple['Var', FMutations]]] = {}

    def items(self) -> t.Iterable[t.Tuple[str, 'DefMap.Value']]:
        return self._lookup.items()

    def copy(self: 'DefMap') -> 'DefMap':
        new = DefMap()
        for k, v in self._lookup.items():
            new._lookup[k] = v.copy()
        for n, aliases in self._aliases.items():
            new._aliases[n] = copy(aliases)

        return new

    def clear(
        self,
        var: 'Var',
        val: t.Union[ast.stmt, ast.ExceptHandler],
        options: Options,
    ) -> None:
        return self.set(var, val, options)

    def set(
        self,
        var: 'Var',
        val: t.Union[ast.stmt, ast.ExceptHandler],
        options: Options,
        aliases: t.Optional[t.Iterable['Var']]=None,
    ) -> None:
        new_als: t.Dict['Var', t.Set[t.Tuple['Var', FMutations]]] = {}
        for k, v in self._aliases.items():
            # This value is overridden
            if var.is_prefix(k):
                continue

            new_als[k] = set()
            for v_new, before in v:
                if var.is_prefix(v_new):
                    continue
                new_als[k].add((v_new, before))

        self._aliases = new_als

        if var.attrs and var.attrs[-1] == '||' and var.base in self._lookup:
            self.mutate(var, val, options)
            return None

        if aliases:
            self._aliases[var] = set(
                (a, frozenset(self.get(a, True))) for a in aliases
                if not var.is_prefix(a)
            )

        if var.attrs and not options.set_props:
            self.mutate(var, val, options)
        elif var.attrs:
            cur = self._lookup[var.base]
            for attr in var.attrs[:-1]:
                cur = cur.submap._lookup[attr]

            new = DefMap.Value({val}, DefMap())
            cur.submap._lookup[var.attrs[-1]] = new
        else:
            self._lookup[var.base] = DefMap.Value({val}, DefMap())

    def get_aliases(self, var: 'Var') -> t.Iterable['Var']:
        for item, _ in var.get_from_map(self._aliases, [], False, set()):
            yield item

    def mutate(
        self,
        var: 'Var',
        val: t.Union[ast.stmt, ast.ExceptHandler],
        options: Options
    ) -> t.Optional['DefMap']:
        if var.base not in self._lookup:
            logging.warn(
                'Variable "{}" can possible not be mutated'.format(var)
            )
            return None

        for alias, _ in var.get_from_map(self._aliases, [], False, set()):
            self.mutate(alias, val, options)

        cur = self._lookup[var.base]
        if var.attrs and options.set_props:
            for attr in var.attrs:
                cur = cur.submap._lookup[attr]
        cur.mutations.add(val)
        return cur.submap

        if not options.mutate_recursive:
            todo = [cur.submap]
            while todo:
                n = todo.pop()
                for _, v in n.items():
                    v.mutations.add(val)
                    if v.submap:
                        todo.append(v.submap)

    def get(
        self,
        var: 'Var',
        recurse: bool,
        done: t.Set['Var']=None,
    ) -> t.Iterable[Mutation]:
        if done is None:
            done = set()

        if var.base in self._lookup:
            cur = self._lookup[var.base]
            for attr in var.attrs:
                if attr not in cur.submap._lookup:
                    return
                cur = cur.submap._lookup[attr]
            yield from cur.get_all_mutations(recurse)

        for alias, before in var.get_from_map(self._aliases, [], recurse, done):
            if alias not in done:
                done.add(alias)
                out = set(self.get(alias, recurse, done))
                yield from list(out - before)

    @staticmethod
    def merge(one: 'DefMap', other: 'DefMap') -> 'DefMap':
        new = one.copy()
        for k, v in other.items():
            new._lookup[k].merge_into(v)

        for n, aliases in other._aliases.items():
            if n in new._aliases:
                new._aliases[n].update(aliases)
            else:
                new._aliases[n] = copy(aliases)

        return new


class DDG:
    def __init__(self) -> None:
        self.mapping: t.DefaultDict[
            ast.stmt, t.Set[t.Union[ast.stmt, ast.ExceptHandler]]
        ] = collections.defaultdict(set)

    def use_var(
        self,
        stmt: ast.stmt,
        var: 'Var',
        def_map: DefMap,
        recurse: bool,
    ) -> None:
        self.mapping[stmt].update(def_map.get(var, recurse))


class Var:
    __slots__ = ['base', 'attrs']

    def __repr__(self) -> str:
        return str(self)

    def __init__(self, base: str) -> None:
        self.base = base
        self.attrs: t.List[str] = []

    def copy(self) -> 'Var':
        new = Var(self.base)
        new.attrs.extend(self.attrs)
        return new

    def index(self) -> 'Var':
        return self.copy()

    def get_from_map(
        self,
        mapping: t.Dict['Var', t.Set[t.Tuple['Var', FMutations]]],
        default: T,
        recurse: bool,
        done: t.Set['Var'],
    ) -> t.Union[t.Set[t.Tuple['Var', FMutations]], T]:
        def create_res(key: 'Var') -> t.Set[t.Tuple['Var', FMutations]]:
            res: t.Set[t.Tuple['Var', FMutations]] = set()
            if key in done:
                return set()
            for v, before in mapping[key]:
                v = v.copy()
                v.attrs.extend(to_add)
                res.add((v, before))
            return res

        to_add: t.List[str] = []

        var = self.copy()
        while var.attrs:
            if var in mapping:
                return create_res(var)
            to_add.append(var.attrs.pop(-1))
        if var in mapping:
            return create_res(var)

        if recurse:
            out: t.Set[t.Tuple['Var', FMutations]] = set()
            for alias in mapping.keys():
                if self.is_prefix(alias):
                    out.update(create_res(alias))
            if out:
                return out

        return default

    def is_prefix(self, other: 'Var') -> bool:
        if self.base != other.base or len(self.attrs) > len(other.attrs):
            return False
        if not self.attrs:
            return True
        return all(attr == other.attrs[i] for i, attr in enumerate(self.attrs))

    def add_attr(self, attr: str) -> 'Var':
        new = self.copy()
        new.attrs.append(attr)
        return new

    def drop_attr(self, index: int=0) -> 'Var':
        new = self.copy()
        if new.attrs:
            del new.attrs[index]
        return new

    def __str__(self) -> str:
        if self.attrs:
            return '{}.{}'.format(self.base, '.'.join(self.attrs))
        else:
            return self.base

    def __hash__(self) -> int:
        return hash('{}.{}'.format(self.base, '.'.join(self.attrs)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return False
        return self.base == other.base and self.attrs == other.attrs


def flatten(iterator: t.Iterable[t.Iterable[T]]) -> t.Iterable[T]:
    for item in iterator:
        yield from item


class Loop:
    def __init__(self) -> None:
        self.break_map = DefMap()
        self.continue_map = DefMap()


def process_block(
    stmts: t.Iterable[ast.stmt],
    loop: t.Optional[Loop],
    def_map: DefMap,
    res_graph: DDG,
    options: Options,
) -> DefMap:
    for stmt in stmts:
        def_map = create_ddg_stmt(
            stmt,
            loop,
            def_map,
            res_graph,
            options,
        )
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return def_map
    return def_map


def create_ddg_try(
    stmt: ast.Try,
    loop: t.Optional[Loop],
    def_map: DefMap,
    res_graph: DDG,
    options: Options,
) -> DefMap:
    handler_maps: t.List[DefMap] = []
    def_map = process_block(stmt.body, loop, def_map, res_graph, options)

    for handler in stmt.handlers:
        handler_map = def_map.copy()
        if handler.name:
            handler_map.set(Var(handler.name), handler, options)
        handler_map = process_block(
            handler.body, loop, handler_map, res_graph, options
        )
        handler_maps.append(handler_map)

    if stmt.orelse:
        def_map = process_block(stmt.orelse, loop, def_map, res_graph, options)

    for handler_map in handler_maps:
        def_map = DefMap.merge(def_map, handler_map)

    if stmt.finalbody:
        def_map = process_block(
            stmt.finalbody, loop, def_map, res_graph, options
        )

    return def_map


def create_ddg_class(
    cls: t.Union[ast.Module, ast.ClassDef], res_graph: DDG, options: Options
) -> None:
    top_fun_stmts: t.List[ast.stmt] = []

    for stmt in cls.body:
        if isinstance(stmt, ast.ClassDef):
            create_ddg_class(stmt, res_graph, options)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            create_ddg_stmt(stmt, None, DefMap(), res_graph, options)
        else:
            top_fun_stmts.append(stmt)

    if top_fun_stmts:
        create_ddg_stmt(
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
            ),
            None,
            DefMap(),
            res_graph,
            options,
        )


def create_ddg_loop(
    loop: t.Union[ast.While, ast.For, ast.AsyncFor],
    outer_loop: t.Optional[Loop],
    def_map: DefMap,
    res_graph: DDG,
    options: Options,
) -> DefMap:
    old_map = def_map.copy()

    loop_st = Loop()
    for index in range(2):
        if isinstance(loop, ast.While):
            create_expr_ddg(
                loop.test, def_map, res_graph, loop, options, False
            )
        else:
            if index == 0:
                create_expr_ddg(
                    loop.iter, def_map, res_graph, loop, options, False
                )
            for target in create_expr_ddg(
                loop.target, def_map, res_graph, loop, options, False
            ):
                def_map.set(target, loop, options)

        if index == 0:
            with no_warn():
                def_map = process_block(
                    loop.body, loop_st, def_map, res_graph, options
                )
        else:
            def_map = process_block(
                loop.body, loop_st, def_map, res_graph, options
            )

        def_map = DefMap.merge(def_map, loop_st.continue_map)

    def_map = DefMap.merge(def_map, old_map)
    if loop.orelse:
        def_map = process_block(
            loop.orelse, outer_loop, def_map, res_graph, options
        )

    return DefMap.merge(def_map, loop_st.break_map)


def create_ddg_stmt(
    stmt: ast.stmt,
    loop: t.Optional[Loop],
    def_map: DefMap,
    res_graph: DDG,
    options: Options,
) -> DefMap:
    def do_expr(expr: ast.expr) -> t.Iterable[Var]:
        return create_expr_ddg(expr, def_map, res_graph, stmt, options)

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        new_def_map = DefMap()
        for arg in chain(stmt.args.args, stmt.args.kwonlyargs):
            new_def_map.set(Var(arg.arg), stmt, options)
        if stmt.args.vararg:
            new_def_map.set(Var(stmt.args.vararg.arg), stmt, options)
        if stmt.args.kwarg:
            new_def_map.set(Var(stmt.args.kwarg.arg), stmt, options)
        process_block(
            stmt.body,
            None,
            new_def_map,
            res_graph,
            options,
        )
        return def_map
    elif isinstance(stmt, ast.ClassDef):
        create_ddg_class(stmt, res_graph, options)
        return def_map
    elif isinstance(stmt, ast.Continue):
        assert loop is not None
        loop.continue_map = DefMap.merge(loop.continue_map, def_map)
        return DefMap()
    elif isinstance(stmt, ast.Break):
        assert loop is not None
        loop.break_map = DefMap.merge(loop.break_map, def_map)
        return DefMap()
    elif isinstance(stmt, ast.Return):
        if stmt.value:
            do_expr(stmt.value)
        return DefMap()
    elif isinstance(stmt, ast.Raise):
        if stmt.exc:
            do_expr(stmt.exc)
        return def_map
    elif isinstance(stmt, ast.Assert):
        do_expr(stmt.test)
        return def_map
    elif isinstance(stmt, ast.Delete):
        deleted = flatten(do_expr(target) for target in stmt.targets)
        for delete in deleted:
            def_map.clear(delete, stmt, options)
        return def_map
    elif isinstance(
        stmt,
        ast.Assign,
    ):
        vals = list(do_expr(stmt.value))
        targets = flatten(do_expr(target) for target in stmt.targets)
        for target in targets:
            def_map.set(target, stmt, options, vals)
        return def_map
    elif isinstance(
        stmt,
        ast.AnnAssign,  # type: ignore
    ):
        if stmt.value:
            do_expr(stmt.value)
        targets = do_expr(stmt.target)
        if not stmt.value:
            return def_map
        for target in targets:
            def_map.set(target, stmt, options)
        return def_map
    elif isinstance(
        stmt,
        ast.AugAssign,
    ):
        if stmt.value:
            do_expr(stmt.value)
        targets = list(do_expr(stmt.target))
        if not stmt.value:
            return def_map
        for target in targets:
            res_graph.use_var(stmt, target, def_map, True)
            def_map.mutate(target, stmt, options)
        for target in targets:
            def_map.set(target, stmt, options)
        return def_map
    elif isinstance(stmt, ast.Expr):
        do_expr(stmt.value)
        return def_map
    elif isinstance(
        stmt,
        (ast.Import, ast.ImportFrom, ast.Nonlocal, ast.Global, ast.Pass, )
    ):
        return def_map
    elif isinstance(stmt, ast.If):
        do_expr(stmt.test)
        then_map = process_block(
            stmt.body, loop, def_map.copy(), res_graph, options
        )
        else_map = process_block(
            stmt.orelse, loop, def_map.copy(), res_graph, options
        )

        return DefMap.merge(then_map, else_map)
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return create_ddg_loop(stmt, loop, def_map, res_graph, options)
    elif isinstance(stmt, (ast.AsyncWith, ast.With)):
        for with_item in stmt.items:
            do_expr(with_item.context_expr)
            if with_item.optional_vars:
                for target in do_expr(with_item.optional_vars):
                    def_map.set(target, stmt, options)
        return process_block(stmt.body, loop, def_map, res_graph, options)
    elif isinstance(stmt, ast.Try):
        return create_ddg_try(stmt, loop, def_map, res_graph, options)
    else:
        assert False


def create_slice_ddg(
    s: ast.slice,
    def_map: DefMap,
    res_graph: DDG,
    cur_stmt: ast.stmt,
    options: Options,
) -> None:
    if isinstance(s, ast.Index):
        create_expr_ddg(s.value, def_map, res_graph, cur_stmt, options, True)
    elif isinstance(s, ast.ExtSlice):
        for dim in s.dims:
            create_slice_ddg(dim, def_map, res_graph, cur_stmt, options)
    elif isinstance(s, ast.Slice):
        if s.lower:
            create_expr_ddg(
                s.lower, def_map, res_graph, cur_stmt, options, True
            )
        if s.upper:
            create_expr_ddg(
                s.upper, def_map, res_graph, cur_stmt, options, True
            )
        if s.step:
            create_expr_ddg(
                s.step, def_map, res_graph, cur_stmt, options, True
            )
    else:
        assert False


def create_expr_ddg(
    expr: ast.expr,
    def_map: DefMap,
    res_graph: DDG,
    cur_stmt: ast.stmt,
    options: Options,
    recurse_name: bool=True,
) -> t.Iterable[Var]:
    def recurse(e: ast.expr, rec: bool=recurse_name) -> t.Iterable[Var]:
        return create_expr_ddg(
            e,
            def_map,
            res_graph,
            cur_stmt,
            options,
            rec,
        )

    res: t.Iterable[Var]

    if isinstance(expr, ast.BoolOp):
        return flatten([recurse(val) for val in expr.values])
    elif isinstance(expr, ast.BinOp):
        res = chain(recurse(expr.left), recurse(expr.right))
        if not options.pure_operators:
            for item in res:
                def_map.mutate(item, cur_stmt, options)
        # No name can be returned here
        return []
    elif isinstance(expr, ast.UnaryOp):
        res = recurse(expr.operand)
        if not options.pure_operators:
            for item in res:
                def_map.mutate(item, cur_stmt, options)
        return []
    elif isinstance(expr, ast.Lambda):
        logging.warning('Lambda\'s are not yet supported')
        return []
    elif isinstance(expr, ast.IfExp):
        recurse(expr.test)
        return chain(recurse(expr.body), recurse(expr.orelse))
    elif isinstance(expr, ast.Dict):
        for k in expr.keys:
            if k is not None:
                recurse(k)
        return flatten(list(map(recurse, expr.values)))
    elif isinstance(
        expr,
        (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
    ):
        # TODO: Do something here!
        return []
    elif isinstance(
        expr,
        ast.Starred,
    ):
        return recurse(expr.value)
    elif isinstance(
        expr,
        (
            ast.Yield,
            ast.YieldFrom,
            ast.Await,
            ast.FormattedValue,  # type: ignore
        ),
    ):
        if expr.value:
            return recurse(expr.value)
        else:
            return []
    elif isinstance(expr, ast.Compare):
        res = recurse(expr.left)
        if not options.pure_operators:
            for item in res:
                def_map.mutate(item, cur_stmt, options)
        for new in map(recurse, expr.comparators):
            if not options.pure_operators:
                for item in new:
                    def_map.mutate(item, cur_stmt, options)
            res = chain(res, new)
        return res
    elif isinstance(expr, ast.Call):
        funs = recurse(expr.func)
        args = chain(
            flatten([recurse(arg) for arg in expr.args]),
            flatten([recurse(k.value) for k in expr.keywords])
        )
        if not options.method_args:
            for arg in args:
                def_map.mutate(arg, cur_stmt, options)

        # New loop `a` is in funs twice.
        for fun in funs:
            if fun.attrs:
                def_map.mutate(fun.drop_attr(-1), cur_stmt, options)
            for alias in def_map.get_aliases(fun):
                if alias.attrs:
                    def_map.mutate(alias.drop_attr(-1), cur_stmt, options)

        return []
    elif isinstance(expr, ast.Name):
        var = Var(expr.id)
        if isinstance(expr.ctx, ast.Load):
            res_graph.use_var(cur_stmt, var, def_map, recurse_name)
        return [var]
    elif isinstance(
        expr,
        (
            ast.Num,
            ast.Bytes,
            ast.Str,
            ast.NameConstant,
            ast.Ellipsis,
            ast.Constant,  # type: ignore
        )
    ):
        return []
    elif isinstance(
        expr,
        ast.JoinedStr,  # type: ignore
    ):
        return flatten([recurse(val) for val in expr.values])
    elif isinstance(expr, ast.Attribute):
        vals = [v.add_attr(expr.attr) for v in recurse(expr.value, False)]
        if isinstance(expr.ctx, (ast.Load)):
            for v in vals:
                res_graph.use_var(cur_stmt, v, def_map, recurse_name)
        elif isinstance(expr.ctx,
                        (ast.Store, ast.Del)) and not options.set_props:
            for v in vals:
                v_prev = v.drop_attr(-1)
                def_map.mutate(v_prev, cur_stmt, options)

        if not options.access_props:
            for v in vals:
                def_map.mutate(v.drop_attr(-1), cur_stmt, options)

        return vals
    elif isinstance(expr, ast.Subscript):
        val = recurse(expr.value, False)
        create_slice_ddg(expr.slice, def_map, res_graph, cur_stmt, options)
        if not options.pure_operators:
            for item in val:
                def_map.mutate(item, cur_stmt, options)

        res = [v.add_attr('||') for v in val]
        if isinstance(expr.ctx, ast.Load):
            for v in res:
                res_graph.use_var(cur_stmt, v, def_map, recurse_name)

        return res
    elif isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return flatten([recurse(elt) for elt in expr.elts])
    else:
        print(expr)
        assert False


def create_ddg(tree: ast.Module) -> DDG:
    res_graph = DDG()
    options = Options(
        access_props=True,
        method_args=True,
        pure_operators=True,
        mutate_recursive=True,
        set_props=True,
    )
    create_ddg_class(tree, res_graph, options)
    return res_graph
