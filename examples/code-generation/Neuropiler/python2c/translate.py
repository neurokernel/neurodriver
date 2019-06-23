#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import re
import ast

from . import blocks

from .block_utils import *
from collections import OrderedDict

def prettyparseprintfile(filename, spaces=4):
    with open(filename, "r") as f:
        prettyparseprint(f.read(), spaces)


def prettyparseprint(code, spaces=4):
    node = ast.parse(code)
    text = ast.dump(node)
    indent_count = 0
    i = 0
    while i < len(text):
        c = text[i]

        if text[i:i+2] in ("()", "[]"):
            i += 1
        elif c in "([":
            indent_count += 1
            indentation = spaces*indent_count
            text = text[:i+1] + "\n" + " "*indentation + text[i+1:]
        elif c in ")]":
            indent_count -= 1
            indentation = spaces*indent_count
            text = text[:i] + "\n" + " "*indentation + text[i:]
            i += 1 + indentation

            if text[i:i+3] in ("), ", "], "):
                text = text[:i+2] + "\n" + " "*indentation + text[i+3:]
                i += indentation

        i += 1
    print(text)


def includes_from_code(code):
    """
    Return a list of #includes that are necessary to run the
    C code.
    """
    includes = []
    # if any("print" in line for line in code):
    #     includes.append(blocks.StringBlock("#include <stdio.h>"))
    #     includes.append(blocks.StringBlock('#include "c_utils/utils.h"'))
    includes.append(blocks.StringBlock('#define EXP exp%(fletter)s'))
    includes.append(blocks.StringBlock('#define POW pow%(fletter)s'))
    includes.append(blocks.StringBlock('#define ABS fabs%(fletter)s'))
    # Add a blank line for no reason
    includes.append(blocks.StringBlock())
    return includes


def main_function(updates = ['spike_state', 'V'],
                  accesses = ['I'],
                  params = ['resting_potential', 'threshold', 'reset_potential', 'capacitance', 'resistance'],
                  internals = OrderedDict([('internalV',0.0)]),
                  localvars = []):
    """
    Return a standard main function block for given Neurodriver variables.
    """
    arg_blocks = [
            blocks.ExprBlock("int", "num_comps", is_arg=True),
            blocks.ExprBlock("%(dt)s", "dt", is_arg=True),
            blocks.ExprBlock("int", "n_steps", is_arg=True)]
    

    template_blocks = [blocks.StringBlock("int tid = threadIdx.x + blockIdx.x * blockDim.x;"),
                       blocks.StringBlock("int total_threads = gridDim.x * blockDim.x;"),
                       blocks.StringBlock("%(dt)s ddt = dt*1000.; // s to ms"),]

    end_blocks = [blocks.StringBlock()]

    for i in accesses:
        arg_blocks.append(blocks.ExprBlock("\n%(" + i + ")s", "g_" + i, pointer_depth=1, is_arg=True))
        template_blocks.append(blocks.StringBlock("%(" + i + ")s " + i + ";"))
    for i in params:
        arg_blocks.append(blocks.ExprBlock("\n%(" + i + ")s", "g_" + i, pointer_depth=1, is_arg=True))
        template_blocks.append(blocks.StringBlock("%(" + i + ")s " + i + ";"))
    for i in internals:
        arg_blocks.append(blocks.ExprBlock("\n%(" + i + ")s", "g_" + i, pointer_depth=1, is_arg=True))
        template_blocks.append(blocks.StringBlock("%(" + i + ")s " + i + ";"))
    for i in updates:
        arg_blocks.append(blocks.ExprBlock("\n%(" + i + ")s", "g_" + i, pointer_depth=1, is_arg=True))
        template_blocks.append(blocks.StringBlock("%(" + i + ")s " + i + ";"))
    for i in localvars:
        template_blocks.append(blocks.StringBlock("%(" + "dt" + ")s " + i + ";"))

    template_blocks.append(blocks.StringBlock("for(int i_comp = tid; i_comp < num_comps; i_comp += total_threads) {"))
    if 'spike_state' in updates:
        template_blocks.append(blocks.StringBlock("spike = 0;"))
    for i in accesses:
        template_blocks.append(blocks.StringBlock(i + " = g_" + i + "[i_comp];"))
    for i in params:
        template_blocks.append(blocks.StringBlock(i + " = g_" + i + "[i_comp];"))
        end_blocks.append(blocks.StringBlock("g_" + i + "[i_comp] = " + i + ";"))
    for i in internals:
        template_blocks.append(blocks.StringBlock(i + " = g_" + i + "[i_comp];"))
        end_blocks.append(blocks.StringBlock("g_" + i + "[i_comp] = " + i + ";"))
    for i in updates:
        template_blocks.append(blocks.StringBlock(i + " = g_" + i + "[i_comp];"))
        end_blocks.append(blocks.StringBlock("g_" + i + "[i_comp] = " + i + ";"))

    end_blocks.append(blocks.StringBlock("}"))

    main_block = blocks.FunctionBlock(
        "__global__ void", "update", arg_blocks, sticky_front=template_blocks,sticky_end=end_blocks
    )
    return main_block


def should_ignore_line(line):
    """
    Determine whether or not a line should be ignored for now.
    """
    patterns_to_ingore = [
        re.compile('#!/usr/bin/env python'),
        re.compile('from __future__ import .+'),
        re.compile('^#[\s\S]*'),
        re.compile('^\s+$')
    ]
    return any(p.search(line) for p in patterns_to_ingore)


def should_keep_line(line):
    """
    Just the oposite of should_ignore_line.
    Wanted to use should_ignore_line in a filter function, but
    coudln't do 'code = filter(not should_ignore_line, code)'
    """
    return not should_ignore_line(line)


def filter_body_nodes(body):
    ignored_nodes = [ast.ImportFrom]
    nodes = []
    for node in body:
        if any(map(lambda x: isinstance(node, x), ignored_nodes)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func.ctx, ast.Load):
                if node.value.func.id != "print":
                    continue
        nodes.append(node)
    return nodes


def get_op(op, arg1, arg2=None):
    if isinstance(op, ast.Add):
        return "(({}) + ({}))".format(arg1, arg2)
    elif isinstance(op, ast.Sub):
        return "(({} - {}))".format(arg1, arg2)
    elif isinstance(op, ast.Mult):
        return "(({}) * ({}))".format(arg1, arg2)
    elif isinstance(op, ast.Div) or isinstance(op, ast.FloorDiv):
        return "(({}) / ({}))".format(arg1, arg2)
    elif isinstance(op, ast.Mod):
        return "(({}) % ({}))".format(arg1, arg2)
    elif isinstance(op, ast.Pow):
        return "POW((double){}, (double){})".format(arg1, arg2)
    elif isinstance(op, ast.LShift):
        return "(({}) << ({}))".format(arg1, arg2)
    elif isinstance(op, ast.RShift):
        return "(({}) >> ({}))".format(arg1, arg2)
    elif isinstance(op, ast.BitOr):
        return "(({}) | ({}))".format(arg1, arg2)
    elif isinstance(op, ast.BitXor):
        return "(({}) ^ ({}))".format(arg1, arg2)
    elif isinstance(op, ast.BitAnd):
        return "(({}) & ({}))".format(arg1, arg2)
    elif isinstance(op, ast.USub):
        if isinstance(arg1,ast.BinOp):
            return "(-{})".format(handle_op_node(arg1))
        else:
            return "(-{})".format(arg1.id)
    elif isinstance(op, ast.UAdd):
        if isinstance(arg1,ast.BinOp):
            return "(+{})".format(handle_op_node(arg1))
        else:
            return "(+{})".format(arg1.id)
    raise Exception("Could not identify operator " + str(op))


def handle_op_node(node):
    if isinstance(node, ast.BinOp):
        if isinstance(node.left, ast.Num):
            node_left = node.left.n
        if isinstance(node.left, ast.Name):
            node_left = node.left.id
        if isinstance(node.left, ast.BinOp):
            node_left = handle_op_node(node.left)
        if isinstance(node.left, ast.UnaryOp):
            node_left = handle_op_node(node.left)
        if isinstance(node.left, ast.Call):
            arguments = '(' + ','.join([str(handle_op_node(i)) for i in node.left.args]) + ')'
            node_left = node.left.func.id+'%(fletter)s' + arguments
        else:
            print(node.left)
            # print('Warning: Could not find the type of left node ', node.left,' in a BinOp. Translation might be incorrect.')

        if isinstance(node.right, ast.Num):
            node_right = node.right.n
        if isinstance(node.right, ast.Name):
            node_right = node.right.id
        if isinstance(node.right, ast.BinOp):
            node_right = handle_op_node(node.right)
        if isinstance(node.right, ast.UnaryOp):
            node_right = handle_op_node(node.right)
        if isinstance(node.right, ast.Call):
            arguments = '(' + ','.join([str(handle_op_node(i)) for i in node.right.args]) + ')'
            node_right = node.right.func.id+'%(fletter)s' + arguments
        else:
            print(node.right)
            # print('Warning: Could not find the type of', node.right,'. Translation might be incorrect.')

        return get_op(node.op, node_left, node_right)
    
    if isinstance(node, ast.UnaryOp):
        return get_op(node.op, node.operand)
    elif isinstance(node, ast.Call):
        arguments = '(' + ','.join([str(handle_op_node(i)) for i in node.args]) + ')'
        return node.func.id +'%(fletter)s'+ arguments
    elif isinstance(node, ast.Compare):
        return handle_op_node(node.left) + ''.join([str(handle_op_node(i)) for i in node.ops]) + ''.join([str(handle_op_node(i)) for i in node.comparators])
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Gt):
        return '>'
    elif isinstance(node, ast.Eq):
        return '=='
    elif isinstance(node, ast.NotEq):
        return '!='
    elif isinstance(node, ast.Lt):
        return '<'
    elif isinstance(node, ast.LtE):
        return '<='
    elif isinstance(node, ast.GtE):
        return '>='
    elif isinstance(node, ast.Is):
        return '='
    elif isinstance(node, ast.IsNot):
        return '!='
    elif isinstance(node, ast.In):
        return '<'
    elif isinstance(node, ast.NotIn):
        return '>='
    elif isinstance(node, ast.And):
        return '&&'
    elif isinstance(node, ast.Or):
        return '||'
    elif isinstance(node, ast.BoolOp):
        return (handle_op_node(node.op)).join(['(' + str(handle_op_node(i))+ ')' for i in node.values])
    # print(node.left, node.right)
    raise Exception("Could not identify node op")


def evaluate_node(node, parent):
    """
    Given a node, evaluate it and add a result to the parent node.
    """
    if isinstance(node, ast.For):
        iterator = node.target.id

        # Add unique iterator (int) that may be reused
        iterator_block = blocks.ExprBlock("int", "iter_" + str(iterator))
        if iterator_block not in parent.variables:
            parent.append_block(iterator_block)
        if node.iter.func.id == "range":
            # Create a new list to be immediately used then destroyed.
            # First find the appropriate parameters for the C range func.
            if len(node.iter.args) == 1:
                start = 0
                stop = handle_op_node(node.iter.args[0])
                step = 1
            elif len(node.iter.args) == 2:
                start = handle_op_node(node.iter.args[0])
                stop = handle_op_node(node.iter.args[1])
                step = 1
            elif len(node.iter.args) == 3:
                start = handle_op_node(node.iter.args[0])
                stop = handle_op_node(node.iter.args[1])
                step = handle_op_node(node.iter.args[2])
            else:
                raise Exception(
                    "Invalid number of arguments found for range")

            # Create the range
            range_obj = blocks.AssignBlock(
                "Object", "temp_range_list"+str(len(parent.variables)),
                "range({},{},{})".format(start, stop, step),
                pointer_depth=1)

            # Create the getter object for the iterator.
            # This does not need ot be freed since we are just
            # redirecting a pointer.
            num_obj = blocks.AssignBlock(
                "Object", iterator,
                "list_get({}, {})"
                .format(range_obj.name, iterator_block.name),
                pointer_depth=1)

            # Create the loop, and put the range constructor before it
            # and the range destructor after it.
            range_block = blocks.ForBlock(
                iterator_block.name, stop,
                before=[], after=[range_obj.destructor()],
                sticky_front=[])

            # Add the for loop to the parent block
            parent.append_block(range_block)

            # Add the contents of the body of the for loop.
            # Filter for unecessary lines first.
            for_body = filter_body_nodes(node.body)
            for f_node in for_body:
                evaluate_node(f_node, range_block)
    elif isinstance(node, ast.If):
        range_block = blocks.IfBlock(
                handle_op_node(node.test),
                before=[], after=[],
                sticky_front=[])
        parent.append_block(range_block)
        for_body = filter_body_nodes(node.body)
        print("If Statement: ", node.body)
        for f_node in node.body:
            print(f_node)
            evaluate_node(f_node, range_block)
    elif isinstance(node, ast.Expr):
        print('expr')
        if isinstance(node.value, ast.Call):
            if node.value.func.id == "print":
                arguments = node.value.args
                if len(arguments) == 1:
                    arg = arguments[0]
                    parent.append_block(blocks.PrintBlock(arg))
    elif isinstance(node, ast.Assign):
        targets = node.targets
        value = node.value

        # prettyparseprint(value.left)
        if is_literal(value):
            if isinstance(value, ast.Num):
                var = value.n
                for target in targets:
                    # Make sure the target is a variable
                    # and you are storing a value
                    assert isinstance(target, ast.Name)
                    assert isinstance(target.ctx, ast.Store)
                    num_obj = blocks.AssignBlock(
                        "Object", target.id, "{}".format(var),
                        pointer_depth=1)
                    parent.append_block(num_obj)
                    # parent.prepend_sticky_end(num_obj.destructor())
            else:
                raise Exception(
                    "No support yet for loading a value from literal {}"
                    .format(value))
        elif isinstance(value, ast.BinOp):
            for target in targets:
                num_obj = blocks.AssignBlock(
                            "Object", target.id, handle_op_node(value),
                            pointer_depth=1)
                # print(num_obj)
                parent.append_block(num_obj)
        # elif isinstance(value, ast.UnaryOp):
        #     print(value.n,'is unary')
        else:
            print("Is BinOp?", isinstance(value, ast.BinOp))
            print(handle_op_node(value))
            for target in targets:
                num_obj = blocks.AssignBlock(
                            "Object", target.id, "{}".format(handle_op_node(value)),
                            pointer_depth=1)
                parent.append_block(num_obj)
            # raise Exception(
            #     "No support yet for loading a value from a non-literal")

def translate(file_, indent_size=4, main_func = None):
    """
    The function for actually translating the code.
    code:
        List containing lines of code.
    """
    # Setup
    with open(file_, "r") as f:
        text = f.read()
        text = text.replace("dt","ddt")
        nodes = ast.parse(text).body
        code = text.splitlines()
    blocks.Block.indent = indent_size
    top = blocks.Block(should_indent=False)

    # Run filtering process
    code = filter(should_keep_line, code)

    # Include includes
    top.append_blocks(includes_from_code(code))

    # Add main function
    if main_func is None:
        main_func = main_function()

    top.append_block(main_func)

    nodes = filter_body_nodes(nodes)
    for node in nodes:
        evaluate_node(node, main_func)

    return str(top)

def import_block(top=None,model_base=None):
    if model_base=='BaseAxonHillockModel':
        top.append_block(blocks.StringBlock('from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import *'))
    elif model_base=='BaseSynapseModel':
        top.append_block(blocks.StringBlock('from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import *'))
        top.append_block(blocks.StringBlock())
    top.append_block(blocks.StringBlock())

def pre_run_block(top=None,assign=None,ind=0,indent_size=4):
    top.append_block(blocks.StringBlock('def pre_run(self,update_pointers):',ind))
    ind+=indent_size
    init = 'initV'
    top.append_block(blocks.StringBlock('if \''+str(init)+'\' in self.params_dict:',ind))
    ind+=indent_size
    top.append_block(blocks.StringBlock('self.add_initializer(\'initV\', \'V\', update_pointers)',ind))
    top.append_block(blocks.StringBlock('self.add_initializer(\'initV\', \'internalV\', update_pointers)',ind))
    ind-=indent_size
    if '\'resting_potential\'' in '\n'.join(assign):
        top.append_block(blocks.StringBlock('else:',ind))
        ind+=indent_size
        init = 'resting_potential'
        top.append_block(blocks.StringBlock('self.add_initializer(\'resting_potential\', \'V\', update_pointers)',ind))
        top.append_block(blocks.StringBlock('self.add_initializer(\'resting_potential\', \'internalV\', update_pointers)',ind))
        ind-=indent_size
    ind-=indent_size
    top.append_block(blocks.StringBlock())

def wrapper(file_, indent_size=4, model=None, model_base=None, assign=None):
    ind = 0
    with open(file_, "r") as f:
        text = f.read()
    top = blocks.Block(should_indent=False)
    
    import_block(top,model_base)

    top.append_block(blocks.StringBlock('class '+model+'('+model_base+'):'))
    ind+=indent_size
    top.append_block(blocks.StringBlock(('\n'+' '*ind).join(assign),ind))
    top.append_block(blocks.StringBlock())
    
    if '\'V\'' in '\n'.join(assign):
        pre_run_block(top,assign,ind,indent_size)

    top.append_block(blocks.StringBlock('def get_update_template(self):',ind))
    ind+=indent_size
    top.append_blocks([blocks.StringBlock('template = """',ind),
                    blocks.StringBlock(text),
                    blocks.StringBlock('"""',ind),
                    blocks.StringBlock('return template',ind)])
    ind-=indent_size
    top.append_block(blocks.StringBlock())

    return str(top)
