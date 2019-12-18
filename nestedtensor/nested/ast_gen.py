import torch
import ast
import pprint
from ast import *

s = \
"""
import torch
def foo(a, b):
    return torch.add(a, b)
"""

Subscript(value=Name(id='a', ctx=Load()),
slice=Index(value=Num(n=0)), ctx=Load())

def builtin_wrapper(bn_name, args):
    ast_args = [
            arg(arg='a{}'.format(i), annotation=None) for i in range(len(args))]
    ast_arg_names = [
            Name(id='a{}'.format(i), ctx=Load()) for i in range(len(args))]
    node = Module(body=[
        FunctionDef(name='foo',
            args=arguments(args=ast_args,
                # [
                # arg(arg='a', annotation=None),
                # arg(arg='b', annotation=None)], 
                vararg=None, 
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),

                body=[Return(
                    value=Call(
                        func=Attribute(value=Name(id='torch', ctx=Load()),
                        attr=bn_name,
                        ctx=Load()),
                        args = ast_arg_names,
                        # args=[
                        #     Name(id='a', ctx=Load()),
                        #     Name(id='b', ctx=Load())
                        #     ],
                        keywords=[]))],

#            body=[Return(
#                value=BinOp(left=Name(id='a', ctx=Load()),
#                op=Add(),
#                right=Name(id='b', ctx=Load())
#                ))],

                decorator_list=[],
                returns=None)])
    node = ast.fix_missing_locations(node)
    print("HEEE1")
    print(node)
    print("HEEE2")
    co = compile(node, "<ast>", 'exec')
    print("HEEE3")
    print(co)
    print("HEEE4")
    exec(co)
    print("HEEE5")
    print(locals())
    print(type(locals()))
    print(locals().keys())
    print(locals()['foo'](3, 4))
    # jresult =  torch.jit.script(locals()['foo'])
    result = torch._C._jit_script_compile('asdf', ast, _rcb, get_default_args(obj))
    print("DDD")
    return result
