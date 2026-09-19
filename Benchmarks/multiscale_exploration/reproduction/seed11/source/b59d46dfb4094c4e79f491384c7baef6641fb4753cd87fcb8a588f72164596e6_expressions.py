"""Restricted expression trees evaluated identically with NumPy, torch or UFL.

No eval/exec, attributes, imports, indexing or arbitrary function calls.
Coordinates are SI values. Expressions may use x, y, pi, arithmetic and the
listed elementary functions. Comparisons are allowed only to build where().
"""
import ast
import math
from functools import lru_cache

FUNCTIONS={'sin','cos','tanh','exp','log','sqrt','where'}

@lru_cache(maxsize=128)
def parse(expression):
    if not isinstance(expression,str) or len(expression)>4000:raise ValueError('Invalid expression length')
    tree=ast.parse(expression,mode='eval').body
    allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Constant,ast.Name,ast.Load,ast.Call,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.USub,ast.UAdd,ast.Compare,ast.Lt,ast.LtE,ast.Gt,ast.GtE)
    for node in ast.walk(tree):
        if not isinstance(node,allowed):raise ValueError(f'Forbidden syntax: {type(node).__name__}')
        if isinstance(node,ast.Constant) and (isinstance(node.value,bool) or not isinstance(node.value,(int,float)) or not math.isfinite(node.value)):raise ValueError('Only finite numerical constants')
        if isinstance(node,ast.Name) and node.id not in {'x','y','pi'}|FUNCTIONS:raise ValueError(f'Unknown name: {node.id}')
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords):raise ValueError('Unsupported function call')
    return tree

def evaluate(expression,x,y,backend):
    def run(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return {'x':x,'y':y,'pi':math.pi}[n.id]
        if isinstance(n,ast.UnaryOp):return -run(n.operand) if isinstance(n.op,ast.USub) else run(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=run(n.left),run(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            return a**b
        if isinstance(n,ast.Compare):
            if len(n.ops)!=1:raise ValueError('Chained comparisons are not supported')
            a,b=run(n.left),run(n.comparators[0]);op=n.ops[0]
            if isinstance(op,ast.Lt):return a<b
            if isinstance(op,ast.LtE):return a<=b
            if isinstance(op,ast.Gt):return a>b
            return a>=b
        if isinstance(n,ast.Call):
            args=[run(a) for a in n.args]
            if n.func.id=='where':
                if len(args)!=3:raise ValueError('where needs condition and two branches')
                return backend.where(args[0],args[1]+0*x,args[2]+0*x)
            if len(args)!=1:raise ValueError('Elementary functions need one argument')
            return getattr(backend,n.func.id)(args[0]+0*x)
        raise ValueError('Unsupported expression node')
    return run(parse(expression))+0*x

def conductivity(spec,x,y,backend):
    if isinstance(spec,(int,float)):return spec+0*x
    if isinstance(spec,str):return evaluate(spec,x,y,backend)
    if spec['type']=='expression':return evaluate(spec['expression'],x,y,backend)
    if spec['type']=='layers':
        axis=x if spec['axis']=='x' else y
        interfaces=spec['interfaces'];values=spec['values']
        # Coordinates and values are ordered from negative to positive axis.
        k=values[0]+0*x;smoothing=spec.get('smoothing',0.)
        for i,position in enumerate(interfaces):
            if smoothing>0:k=k+(values[i+1]-values[i])*.5*(1+backend.tanh((axis-position)/smoothing))
            else:k=backend.where(axis<=position,k,values[i+1]+0*x)
        return k
    raise ValueError('Unsupported conductivity specification')


@lru_cache(maxsize=128)
def derivative(expression,variable):
    """Symbolic derivative of the restricted language, without a new dependency."""
    def d(n):
        u=lambda a:ast.unparse(a)
        if isinstance(n,ast.Constant):return '0'
        if isinstance(n,ast.Name):return '1' if n.id==variable else '0'
        if isinstance(n,ast.UnaryOp):return ('-' if isinstance(n.op,ast.USub) else '+')+'('+d(n.operand)+')'
        if isinstance(n,ast.BinOp):
            a,b=u(n.left),u(n.right);da,db=d(n.left),d(n.right)
            if isinstance(n.op,ast.Add):return f'({da})+({db})'
            if isinstance(n.op,ast.Sub):return f'({da})-({db})'
            if isinstance(n.op,ast.Mult):return f'({da})*({b})+({a})*({db})'
            if isinstance(n.op,ast.Div):return f'(({da})*({b})-({a})*({db}))/({b})**2'
            if isinstance(n.right,ast.Constant):
                if n.right.value==0:return '0'
                if n.right.value==1:return da
                return f'({b})*({a})**(({b})-1)*({da})'
            return f'({a})**({b})*(({db})*log({a})+({b})*({da})/({a}))'
        if isinstance(n,ast.Call):
            if n.func.id=='where':return f'where({u(n.args[0])},{d(n.args[1])},{d(n.args[2])})'
            a=u(n.args[0]);da=d(n.args[0])
            factor={'sin':f'cos({a})','cos':f'-sin({a})','exp':f'exp({a})','tanh':f'(1-tanh({a})**2)','log':f'1/({a})','sqrt':f'1/(2*sqrt({a}))'}[n.func.id]
            return f'({factor})*({da})'
        raise ValueError('Cannot differentiate this syntax')
    return d(parse(expression))


def manufactured_source(temperature,spec,x,y,backend):
    if isinstance(spec,(int,float)):ke=str(spec)
    elif isinstance(spec,str):ke=spec
    elif spec['type']=='expression':ke=spec['expression']
    elif spec['type']=='layers':
        if spec.get('smoothing',0)>0:raise ValueError('Use an explicit expression for smooth manufactured layers')
        ke=str(spec['values'][-1])
        for i in reversed(range(len(spec['interfaces']))):
            ke=f"where({spec['axis']} <= {spec['interfaces'][i]}, {spec['values'][i]}, {ke})"
    else:raise ValueError('Unsupported manufactured conductivity')
    k=conductivity(spec,x,y,backend);q=0*x
    for axis in ['x','y']:
        dt=derivative(temperature,axis)
        q=q-evaluate(derivative(ke,axis),x,y,backend)*evaluate(dt,x,y,backend)-k*evaluate(derivative(dt,axis),x,y,backend)
    return q
