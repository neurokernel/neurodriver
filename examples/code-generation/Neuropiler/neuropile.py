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
                model_base = node.bases[0].id
                neuropiler_classes[className] = {'assignments': [], 'functions': {},'model_base': model_base}
                for i in node.body:
                    if isinstance(i, ast.Assign):
                        neuropiler_classes[className]['assignments'].append(astunparse.unparse(i).replace('\n',''))
                    if isinstance(i, ast.FunctionDef):
                        neuropiler_classes[className]['functions']['step'] = astunparse.unparse(i.body)
            else:
                methodName = node.name
    return neuropiler_classes


if __name__ == "__main__":
    import neurokernel
    import inspect

    parser = ArgumentParser(description="Translate Python code to C.")
    parser.add_argument("file", help=".py file to translate.")
    arguments = parser.parse_args()
    results = neuropile_main(arguments.file)

    for model_name in results.keys():
        model_base = results[model_name]['model_base']
        path = inspect.getfile(neurokernel).replace('__init__.py','')+'LPU/NDComponents/'+model_base[4:]+'s/'
        if 'step' in results[model_name]['functions']:
        	# write step function
            with open(model_name+"_temp.py", "w") as python_code:
                python_code.write(results[model_name]['functions']['step'])
            # evaluate the assignments
            main_func_block = eval('main_function(' + (','.join(results[model_name]['assignments'])) + ')')
            # translate the iterative update
            translated_code = translate.translate(model_name+"_temp.py", indent_size=4, main_func = main_func_block)
            # write the whole template
            with open(model_name+"_template.c", "w") as cuda_code:
                cuda_code.write(translated_code)
            wrapped_code = translate.wrapper(model_name+"_template.c", indent_size=4, model=model_name, model_base = model_base, assign = results[model_name]['assignments'])
            with open(path+model_name+"_gen.py", "w") as python_code:
                python_code.write(wrapped_code)
