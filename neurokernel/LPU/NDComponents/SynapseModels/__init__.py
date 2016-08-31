import os
import fnmatch

__all__ = []

NDC_dir = os.path.dirname(__file__)
for root, dirnames, filenames in os.walk(NDC_dir):
    mod_imp = False
    for f in fnmatch.filter(filenames,"*.py"):
        if '__init__'!=f[:8] and 'Base'!=f[:4]:
            if root != NDC_dir and not mod_imp:
                __path__.append(root)
                mod_imp = True
            __all__.append(f[:-3])
