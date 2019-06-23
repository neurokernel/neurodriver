#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import shutil
import subprocess
import sys

from . import translate


def which(program):
    """
    Function for checking if an executable exists on a system.
    Thanks:
    http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def memory_check(translated_code):
    """
    Run valgrind to see if there are any errors.
    """
    if not which("valgrind"):
        print("Could not find valgrind to run the memory check.",
              file=sys.stderr)
        return 1

    if not error_check_c(translated_code):
        print("Could not find compile the C to run the memory check.",
              file=sys.stderr)
        return 1

    tmpfilename = "hopefully_there_arent_any_other_files_with_this_name"
    tmpfile = open(tmpfilename + ".c", "w")
    tmpfile.write(translated_code)
    tmpfile.close()

    p = subprocess.Popen("gcc {}.c c_utils/*.c -o {}".format(tmpfilename,
                         tmpfilename), shell=True)
    p.communicate()

    if os.path.exists(tmpfilename):
        print(subprocess.check_output(
            "valgrind --dsymutil=yes --track-origins=yes ./{}"
            .format(tmpfilename).split()), end=""
        )
        os.remove(tmpfilename)
        shutil.rmtree(tmpfilename + ".dSYM")
    else:
        print("Could not generate an executable due to an error.",
              file=sys.stderr)
    os.remove(tmpfilename + ".c")

    return 0


def error_check_c(translated_code, execute=False):
    """
    Check to see if there are any errors by checking the return
    status of gcc after attempting to compile the translated_code.
    """
    tmpfilename = "hopefully_there_arent_any_other_files_with_this_name"
    tmpfile = open(tmpfilename + ".c", "w")
    tmpfile.write(translated_code)
    tmpfile.close()

    p = subprocess.Popen("gcc {}.c c_utils/*.c -o {}".format(tmpfilename,
                         tmpfilename), shell=True)
    p.communicate()

    if os.path.exists(tmpfilename):
        if execute:
            print(subprocess.check_output(["./" + tmpfilename]), end="")
        else:
            print("Successful compilation!")
        os.remove(tmpfilename)
    else:
        print("Could not generate an executable due to an error.",
              file=sys.stderr)
    os.remove(tmpfilename + ".c")

    # Success on 0 (return True)
    return not p.returncode


def error_check_python(filename):
    """
    Check to see if there are any errors by checking the return
    status of python after attempting to interpret the code.
    """
    with open(os.devnull, "w") as devnull:
        p = subprocess.Popen("python {}".format(filename).split(),
                             stdout=devnull)
        p.communicate()
        if p.returncode:
            print("Could not interpret python code due to an error.",
                  file=sys.stderr)

    # Success on 0 (return True)
    return not p.returncode


def get_args():
    """
    Standard arggument parser creator function.
    Import here because only using this module here.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert python code to C code")
    parser.add_argument("file", help=".py file to translate to C.")
    parser.add_argument(
        "-s", "--indent-size", type=int, default=4,
        help="The number of spaces with which to represent each indent."
    )
    parser.add_argument(
        "-c", "--compile-check", default=False, action="store_true",
        help="Instead of printing to stdout, compile the generated code "
        "and see if there are any errors."
    )
    parser.add_argument(
        "-e", "--execute", default=False, action="store_true",
        help="Instead of translating the code and spitting to stdout, "
        "immediately compile and execute the translated code. This does "
        "not generate any files or print to stdout."
    )
    parser.add_argument(
        "-m", "--memory-check", default=False, action="store_true",
        help="Instead of printing to stdout, compile the generated code "
        "and run valgrind on it to see if there are any memory leaks "
        "in the C translation."
    )
    parser.add_argument(
        "-a", "--ast-tree", default=False, action="store_true",
        help="Print the abstract syntax tree of the python code."
    )

    return parser.parse_args()


def main():
    """
    Stages
    1. Setup stuff from arguments
    2. Ignore certain lines of code.
    3. Add necessary includes and main function.
    4. Actual translation.
    """
    args = get_args()
    """
    if not error_check_python(args.file):
        return 1
    elif args.ast_tree:
        translate.prettyparseprintfile(args.file)
        return 0
    """
    translated_code = translate.translate(
        args.file, indent_size=args.indent_size)

    if args.compile_check:
        return 0 if error_check_c(translated_code) else 2
    elif args.execute:
        return 0 if error_check_c(translated_code, True) else 2
    elif args.memory_check:
        return memory_check(translated_code)
    else:
        print(translated_code)

    return 0


if __name__ == "__main__":
    sys.exit(main())
