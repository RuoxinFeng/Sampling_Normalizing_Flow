#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import ast
import builtins
import math
import operator
from typing import Any, Dict

import numpy as np
import pandas as pd


def _mathematical_function_eval(x: str, arg_dict: Dict[str, Any], subscriptable_dict: Dict[str, Any], obj: Any) -> Any:
    if obj is not None and not isinstance(obj, str):  # example_call: df.abs()
        return getattr(obj, x)(*arg_dict["args"], **arg_dict["kwargs"])
    elif isinstance(obj, str):  # example call: np.abs(-1)
        subscriptable_dict = {**subscriptable_dict, "np": np, "pd": pd, "math": math}
        obj = subscriptable_dict.get(obj)
        if obj is not None:
            return getattr(obj, x)(*arg_dict["args"], **arg_dict["kwargs"])
    else:  # example call: abs(-1)
        try:
            func = getattr(builtins, x)
            return func(*arg_dict["args"], **arg_dict["kwargs"])
        except AttributeError:
            ...
    raise ValueError(f"Unknown function '{x}()' occurred in a formula evaluation")


AVAILABLE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Call: _mathematical_function_eval,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
}


def evaluate_formula(formula: str, subscriptable_dict: Dict[str, Any] = {}) -> Any:
    """Evaluates a python expression that is defined in a string.

    Parameters
    ----------
    formula : str
        The expression to be evaluated.
    subscriptable_dict: Dict[str, Any], optional
        A dictionary containing a mapping of strings (as used in the `formula`) to corresponding python objects.
        Example: For the formula "df['column'] * 2", this dictionary needs to map the string "df" to a
        pd.DataFrame that includes the column to be used in this calculation.

    Returns
    -------
    Any
        The result of the evaluation.

    Raises
    ------
    ValueError
        If the formula includes multiple comparisons (e.g. "1 < a < 2").
    ValueError
        If the formula includes an unknown operation.

    Example
    -------
    >>> evaluate_formula("1+1")
    2
    >>> a = [1,0]
    >>> evaluate_formula("a[0]+1", subscriptable_dict = {"a": a})
    2
    """

    def _eval(node: Any) -> Any:
        if not isinstance(node, ast.AST):
            return node
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return _eval(node.value)
        if isinstance(node, ast.Name):
            return _eval(node.id)
        if isinstance(node, ast.Index):
            return _eval(node.value)
        if isinstance(node, ast.Attribute):
            return _eval(node.attr)
        if isinstance(node, ast.Subscript):
            subscriptable = _eval(node.value)
            if subscriptable not in subscriptable_dict:
                raise ValueError(f"Unknown subscript '{subscriptable}' encountered in a formula evaluation.")
            slice = _eval(node.slice)
            return subscriptable_dict[subscriptable][slice]
        if isinstance(node, ast.BinOp):
            _check_operation_available(node.op)
            return AVAILABLE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            _check_operation_available(node.op)
            return AVAILABLE_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            args = [_eval(x) for x in node.args]
            kwargs = {x.arg: _eval(x.value) for x in node.keywords}
            arg_dict = {"args": args, "kwargs": kwargs}
            obj = _eval(node.func.value) if hasattr(node.func, "value") else None
            return _mathematical_function_eval(
                _eval(node.func), arg_dict=arg_dict, subscriptable_dict=subscriptable_dict, obj=obj
            )
        if isinstance(node, ast.Compare):
            if len(node.comparators) > 1 or len(node.ops) > 1:
                raise ValueError(
                    f"Multiple Comparison not supported in a formula evaluation, got comparators {node.comparators} and operations {node.ops}"
                )
            _check_operation_available(node.ops[0])
            return AVAILABLE_OPS[type(node.ops[0])](_eval(node.left), _eval(node.comparators[0]))
        if isinstance(node, ast.List):
            return [_eval(obj) for obj in node.elts]
        raise ValueError(f"Unsupported element of type {type(node)} found in a formula.")

    return _eval(ast.parse(formula, mode="eval"))


def _check_operation_available(operation: ast.AST) -> None:
    if not AVAILABLE_OPS.get(type(operation)):
        raise ValueError(f"Error in a formula evaluation. Operation of type {type(operation)} is not supported.")
