"""Safe formula evaluation using ast level node validation.
use:
    import formula
    r = formula.evaluate("2+sin(x)", {"x":3.14})
"""

import ast
import math

# create a safe and desirable subset of the built-ins
_safe_builtins = {x: eval(x) for x in "False None True abs all any bool complex divmod enumerate filter float hash int iter len list map max min next pow range reversed round set slice sorted str sum tuple zip".split()}

# add a full set of usable math functions and constants
_safe_builtins.update(((x, getattr(math, x)) for x in dir(math) if not x.startswith('_')))

# add an empty "__builtins__" value, so that Python doesn't add one by itself
# see https://docs.python.org/3/library/functions.html#eval
_safe_builtins["__builtins__"] = {}

try:
    import numpy
    _safe_builtins["np"] = _safe_builtins["_np"] = numpy
except ImportError:
    pass

try:
    from . import custom_fns
    _safe_builtins["opsani"] = custom_fns
except ImportError:
    pass

default_valid_nodes = [
    # Notice we don't allow ast.Attribute here. Check for attributes require more ast tree walking.
    ast.BinOp, ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Tuple, ast.List, ast.Dict,
    ast.Constant,
    ast.Mult, ast.Name, ast.Expression, ast.Load, ast.IfExp,
    ast.Subscript, ast.Starred,
    # binary ops
    ast.Invert, ast.Not, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow, ast.LShift,
    # logical  ops
    ast.And, ast.Or,
    # Bits ops
    ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
    # Compare
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    # compatible for python 3.6
    ast.Str, ast.Num, ast.NameConstant
]


def validate(expr, valid_names=[], valid_nodes=[]):
    if not valid_nodes:
        valid_nodes = default_valid_nodes

    assert isinstance(expr, str), "formula's expression argument should of a string type"
    nodes = list(ast.walk(ast.parse(expr, mode='eval')))

    invalid_names = set((n.id for n in nodes if isinstance(n, ast.Name))) - set(valid_names)
    if invalid_names:
        raise ValueError(f"undefined names {invalid_names} in expression")
    invalid_nodes = set((n.__class__ for n in nodes)) - set(valid_nodes)
    if invalid_nodes:
        raise ValueError(f"Invalid terms {invalid_nodes} in expression")


def evaluate(expr, var):
    """
    Safely evaluate an expression from user-defined string, using pre-defined
    library functions (incl. all the useful math functions/const) and symbolic
    variables.

    Args:
        expr: string containing the expression to evaluate (e.g., 'perf*2/cost')
        var: dict with zero or more variables to use (e.g., {'perf':2000,'cost':0.02})

    The evaluation supports constants, all Python operators as well as a select
    subset of safe Python built-ins and the full math module (not prefixed by 'math.')

    Note that vars will shadow any of the standard const/funcs, e.g., if a var 'pi'
    is included in the vars arg, it will shadow the standard math.pi value.
    """

    validate(expr, list(var.keys()) + list(_safe_builtins.keys()))

    try:
        ret = eval(expr, _safe_builtins, var)
    except Exception as e:
        print('Failed to evaluate formula {}. Got exception {}(cause="{}")'.format(expr, type(e).__name__, e))
        raise
    # print('Formula: evaluated {} with {} to {}'.format(expr, var, ret))
    return ret


def evaluatec(expr, var):
    """same as evaluate, but expects a pre-tested and compiled expression"""
    try:
        ret = eval(expr, _safe_builtins, var)
    except Exception as e:
        print('Failed to evaluate formula {}. Got exception {}(cause="{}")'.format(expr, type(e).__name__, e))
        raise
    return ret
