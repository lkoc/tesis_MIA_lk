"""Restricted algebraic expressions shared by the generic PINN and FEM adapters."""
import ast
import math
from functools import lru_cache

FUNCTIONS = {'sin', 'cos', 'tanh', 'exp', 'log', 'sqrt', 'where'}


@lru_cache(maxsize=128)
def parse(expression):
    if not isinstance(expression, str) or len(expression) > 4000:
        raise ValueError('Invalid expression length')
    tree = ast.parse(expression, mode='eval').body
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
               ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
               ast.UAdd, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f'Forbidden syntax: {type(node).__name__}')
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or
                not isinstance(node.value, (int, float)) or not math.isfinite(node.value)):
            raise ValueError('Only finite numerical constants')
        if isinstance(node, ast.Name) and node.id not in {'x', 'y', 'pi'} | FUNCTIONS:
            raise ValueError(f'Unknown name: {node.id}')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or
                node.func.id not in FUNCTIONS or node.keywords):
            raise ValueError('Unsupported function call')
    return tree


def evaluate(expression, x, y, backend):
    def run(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {'x': x, 'y': y, 'pi': math.pi}[node.id]
        if isinstance(node, ast.UnaryOp):
            return -run(node.operand) if isinstance(node.op, ast.USub) else run(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = run(node.left), run(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            return left ** right
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError('Chained comparisons are not supported')
            left, right = run(node.left), run(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt): return left < right
            if isinstance(op, ast.LtE): return left <= right
            if isinstance(op, ast.Gt): return left > right
            return left >= right
        if isinstance(node, ast.Call):
            args = [run(arg) for arg in node.args]
            if node.func.id == 'where':
                if len(args) != 3:
                    raise ValueError('where needs condition and two branches')
                return backend.where(args[0], args[1] + 0 * x, args[2] + 0 * x)
            if len(args) != 1:
                raise ValueError('Elementary functions need one argument')
            return getattr(backend, node.func.id)(args[0] + 0 * x)
        raise ValueError('Unsupported expression node')
    return run(parse(expression)) + 0 * x
