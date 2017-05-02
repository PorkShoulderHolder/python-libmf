import re
import os
import sys


def find_lib_path():
    path = os.environ["LIBMF_OBJ"] if "LIBMF_OBJ" in os.environ else sys.argv[1] if len(sys.argv) > 1 else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and re.match('python-libmf\.cpython(.*)\.so', i):
            return os.path.join(path, i)
