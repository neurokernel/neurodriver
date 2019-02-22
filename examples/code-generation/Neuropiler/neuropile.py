from argparse import ArgumentParser
import ast
import astunparse
from python2c.python2c import *
from python2c.translate import main_function
from collections import OrderedDict


def neuropile_main(file):
    neuropiler_modules = open(file,"r").read()
    neuropiler_modules = ast.parse(neuropiler_modules)
    node = ast.NodeVisitor()
    neuropiler_classes = {}
    for node in ast.walk(neuropiler_modules):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            if isinstance(node, ast.ClassDef):
                className = node.name
                neuropiler_classes[className] = {'assignments': [], 'functions': {}}
                for i in node.body:
                    if isinstance(i, ast.Assign):
                        neuropiler_classes[className]['assignments'].append(astunparse.unparse(i).replace('\n',''))
                    if isinstance(i, ast.FunctionDef):
                        neuropiler_classes[className]['functions']['step'] = astunparse.unparse(i.body)
            else:
                methodName = node.name
    return neuropiler_classes


if __name__ == "__main__":
    parser = ArgumentParser(description="Translate Python code to C.")
    parser.add_argument("file", help=".py file to translate.")
    arguments = parser.parse_args()
    results = neuropile_main(arguments.file)
    for model_name in results.keys():
        if 'step' in results[model_name]['functions']:
            with open(model_name+".py", "w") as python_code:
                python_code.write(results[model_name]['functions']['step'])
            main_func_block = eval('main_function(' + (','.join(results[model_name]['assignments'])) + ')')
            translated_code = translate.translate(model_name+".py", indent_size=4, main_func = main_func_block)
            with open(model_name+"_template.c", "w") as cuda_code:
                cuda_code.write(translated_code)