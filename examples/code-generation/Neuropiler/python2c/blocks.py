#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from .block_utils import *


class Block(object):
    """
    Class for representing a block of code/scope.

    Every child block indicates another indent in
    the returned code.
    """

    indent = 4

    def __init__(self, contents=None, should_indent=True, sticky_front=None,
                 sticky_end=None, before=None, after=None, variables=None):
        """
        contents:
            List of child blocks
        should_indent:
            Whether or not the contents of this block should
            be indented. This should ideally be only False for
            the highest level block (the one with no parent)
            blocks.
        sticky_front:
            Content to always appear at the front of the contents
            of the block.
        sticky_end:
            Content to always appear at the end of the contents
            of the block.
        before:
            List of blocks to appear before an instance of this block.
        after:
            List of blocks to appear after an instance of this block.
        variables:
            List of variables in the scope of this block.
        """
        # Get around the mutable default arguments
        contents = contents or []
        sticky_front = sticky_front or []
        sticky_end = sticky_end or []
        before = before or []
        after = after or []
        variables = variables or []

        assert all(issubclass(child.__class__, Block) for child in contents)
        self.contents = contents
        self.should_indent = should_indent
        self.sticky_end = sticky_end
        self.sticky_front = sticky_front
        self.before = before
        self.after = after
        self.variables = variables

        # Other initialization
        for var in sticky_front + sticky_end:
            if isinstance(var, ExprBlock) and var not in variables:
                self.append_variable(var)

    @property
    def last(self):
        return self.contents[-1]

    def prepend_sticky_end(self, block):
        """
        Add a block to be stuck at the beginning of the sticky_end.
        """
        self.sticky_end = [block] + self.sticky_end

    def append_sticky_end(self, block):
        """
        Add a block to be stuck at the end of sticky_end.
        """
        self.sticky_end.append(block)

    def append_variable(self, var):
        """
        Add a variable name to the list of variables.
        """
        assert isinstance(var, ExprBlock)
        """
        if var in self.variables:
            raise Exception(
                ("Attempted to add variable '{}' in scope of {} when it "
                 "already exists: {}")
                .format(var, self.__class__, map(str, self.variables)))
        """
        self.variables.append(var)

    def append_block(self, block):
        """
        Append another block to this block's contents.
        Add the variables within the scope of this block to the
        list of variables of the child block.
        """
        assert issubclass(block.__class__, Block)

        self.contents.append(block)
        if isinstance(block, StringBlock):
            pass
        elif isinstance(block, ExprBlock):
            self.append_variable(block)
        else:
            # block is an inline block that contains variables
            # which should be added to this block's scope
            if isinstance(block, InlineBlock):
                for var in block.variables:
                    if (isinstance(var, ExprBlock) and
                            var not in self.variables):
                        self.append_variable(var)

            # block is a block that can hold variables
            for var in self.variables:
                if var not in block.variables:
                    block.append_variable(var)
            for var in block.before + block.after:
                if (isinstance(var, ExprBlock) and
                        var not in self.variables):
                    self.append_variable(var)

    def append_blocks(self, blocks):
        """
        Same as append, but for a list of blocks.
        """
        for block in blocks:
            self.append_block(block)

    def block_strings(self):
        """
        Meant to be called by __str__ to actually return the True
        formatted blocks.
        """
        indentation = " "*self.indent if self.should_indent else ""
        child_contents = []

        for content in self.sticky_front + self.contents + self.sticky_end:
            content = content.block_strings()
            if isinstance(content, list):
                for nested_content in content:
                    child_contents.append(indentation + str(nested_content))
            else:
                child_contents.append(indentation + str(content))
        return child_contents

    def __str__(self):
        return "\n".join(self.block_strings())


class FunctionBlock(Block):
    """
    Block for functions.
    func_type:
        Data type to be returned by the func.
        (int, float, etc.)
    name:
        Function name
    args:
        List of tuples containing CArguments.
    contents:
        List of blocks to fill this block with.
    """
    def __init__(self, func_type, name, args, contents=None, sticky_front=None,
                 sticky_end=None, before=None, after=None, variables=None):
        super(FunctionBlock, self).__init__(
            contents=contents, sticky_front=sticky_front,
            sticky_end=sticky_end, before=before, after=after,
            variables=variables
        )
        self.func_type = func_type
        self.args = args
        self.name = name

        # Add the arguments
        for arg in args:
            self.append_variable(arg)

    def block_strings(self):
        """
        Meant to be called by __str__ to actually return the True
        formatted blocks.
        """
        indentation = " "*self.indent if self.should_indent else ""
        # print(", ".join([str(i) for i in self.args]))
        child_contents = list(map(str, self.before))
        child_contents += ["{type} {name}({args}){{".format(
            type=self.func_type, name=self.name,
            args=", ".join([str(i) for i in self.args]))]

        for content in self.sticky_front + self.contents + self.sticky_end:
            content = content.block_strings()
            if isinstance(content, list):
                for nested_content in content:
                    child_contents.append(indentation + str(nested_content))
            else:
                child_contents.append(indentation + str(content))
        child_contents += ["}"]

        child_contents += map(str, self.after)

        return child_contents

class IfBlock(Block):
    """
    Block for if statements
    """
    def __init__(self, statement, contents=None,
                 sticky_front=None, sticky_end=None, before=None, after=None,
                 variables=None):
        super(IfBlock, self).__init__(
            contents=contents, sticky_front=sticky_front,
            sticky_end=sticky_end, before=before, after=after,
            variables=variables
        )
        self.statement = statement

    def block_strings(self):
        indentation = " "*self.indent if self.should_indent else ""

        child_contents = list(map(str, self.before))
        child_contents += [
            "if ({iterator}){{"
            .format(iterator=self.statement)
        ]

        for content in self.sticky_front + self.contents + self.sticky_end:
            content = content.block_strings()
            if isinstance(content, list):
                for nested_content in content:
                    child_contents.append(indentation + str(nested_content))
            else:
                child_contents.append(indentation + str(content))
        child_contents += ["}"]
        child_contents += map(str, self.after)

        return child_contents


class ForBlock(Block):
    """
    Block for for loops
    """
    def __init__(self, iterator, max_iteration, contents=None,
                 sticky_front=None, sticky_end=None, before=None, after=None,
                 variables=None):
        super(ForBlock, self).__init__(
            contents=contents, sticky_front=sticky_front,
            sticky_end=sticky_end, before=before, after=after,
            variables=variables
        )
        self.iterator = iterator
        self.max_iteration = max_iteration

    def block_strings(self):
        indentation = " "*self.indent if self.should_indent else ""

        child_contents = list(map(str, self.before))
        child_contents += [
            "for({iterator} = 0; {iterator} < {max_iteration}; {iterator}++){{"
            .format(iterator=self.iterator, max_iteration=self.max_iteration)
        ]

        for content in self.sticky_front + self.contents + self.sticky_end:
            content = content.block_strings()
            if isinstance(content, list):
                for nested_content in content:
                    child_contents.append(indentation + str(nested_content))
            else:
                child_contents.append(indentation + str(content))
        child_contents += ["}"]
        child_contents += map(str, self.after)

        return child_contents


class InlineBlock(Block):
    """
    Class representing blocks that
    - are not indented
    - do not contain a body
    """

    def __init__(self, contents=None, before=None, after=None, variables=None):
        super(InlineBlock, self).__init__(
            contents=contents, sticky_front=None, should_indent=False,
            sticky_end=None, before=before, after=after,
            variables=variables
        )


class ExprBlock(InlineBlock):
    """
    Class for specifically declaring a variable.
    """

    def __init__(self, data_type, name, pointer_depth=0, array_depth=0,
                 is_arg=False):
        """
        data_type:
            Data type (int, float, some typedef, etc.)
        name:
            Name of the argument
        pointer_depth:
            Essentially number of pointers to put to the left of the
            argument name when printing.
        array_depth:
            Essentially number of bracket pairs to put to the right of the
            argument name when printing.
        is_arg:
            If this block is an argument to a function, in which case, there
            should be no semicolon in the str representation of this.
        """
        # TODO: Add support for const and other qualifiers later.
        super(ExprBlock, self).__init__()
        self.data_type = data_type
        self.name = name
        self.pointer_depth = pointer_depth
        self.array_depth = array_depth
        self.is_arg = is_arg

    def block_strings(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return "{}{} {}{}{}".format(
            self.data_type, "* "*self.pointer_depth, self.name,
            "[]"*self.array_depth, "" if self.is_arg else ";")


class AssignBlock(ExprBlock):
    """
    Class for assigning a variable.
    """

    def __init__(self, data_type, name, value, pointer_depth=0, array_depth=0):
        """
        data_type:
            Data type (int, float, some typedef, etc.)
        name:
            Name of the argument
        value:
            The right side of the equals sign.
        pointer_depth:
            Essentially number of pointers to put to the left of the
            argument name when printing.
        array_depth:
            Essentially number of bracket pairs to put to the right of the
            argument name when printing.
        """
        super(AssignBlock, self).__init__(
            data_type=data_type, name=name, pointer_depth=pointer_depth,
            array_depth=array_depth
        )
        self.value = value

    def destructor(self):
        if self.data_type == "Object":
            return StringBlock("")
        else:
            raise Exception(("Attempting to call destroy on a variable that"
                             "isn't an object."))

    def __str__(self):
        return "{}{}{}{} = {};".format(
            "", "", self.name,
            "[]"*self.array_depth, self.value)


class PrintBlock(InlineBlock):
    """
    Class for specifically printing a node.
    """

    def __init__(self, node):
        if isinstance(node, ast.Name):
            # Node is a variable
            # Call the str() function, then immediately free it.
            var = node.id
            self.lines = [
                AssignBlock(
                    "char", "{}_str".format(var), "str({})".format(var),
                    pointer_depth=1),
                StringBlock('printf("%s\\n", {}_str);'.format(var)),
                StringBlock("")
            ]
            super(PrintBlock, self).__init__(variables=[self.lines[0]])
        elif is_literal(node):
            if isinstance(node, ast.Str):
                # Node is a string literal
                var = node.s
            elif isinstance(node, ast.Num):
                # Node is a literal number
                var = node.n
            elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
                print(list(node.elts))
                raise Exception(
                    "No support yet for node of type List or Tuple")
            else:
                raise Exception(
                    "No support for the literal node of type {}"
                    .format(node.__class__))

            self.lines = [StringBlock('printf("{}\\n");'.format(var))]
            super(PrintBlock, self).__init__()
        else:
            raise Exception(
                "No support for printing node of type {}"
                .format(node.__class__))

    def block_strings(self):
        return self.lines

    def __str__(self):
        return "\n".join(self.block_strings())


class StringBlock(InlineBlock):
    """
    Block for representing a single line/string from code.
    """
    def __init__(self, contents="",indent=0):
        assert isinstance(contents, str)
        self.contents = contents
        self.indent = indent

    def block_strings(self):
        return " "*self.indent+str(self.contents)

    def __str__(self):
        return " "*self.indent+str(self.contents)
