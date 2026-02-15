import sys
import platform
import subprocess
from setuptools import setup, Extension
import numpy  # <--- Ensure this is imported

def get_brew_prefix():
    try:
        return subprocess.check_output(['brew', '--prefix'], text=True).strip()
    except:
        return "/usr/local"

# Base compiler flags
c_args = ['-O3']
l_args = []

# --- PLATFORM SPECIFIC CONFIGURATION ---
if platform.system() == 'Darwin':
    # macOS
    prefix = get_brew_prefix()
    
    # FIX IS HERE: We need BOTH Numpy headers AND OpenMP headers
    include_dirs = [
        numpy.get_include(),              # <--- 1. NumPy Path
        f'{prefix}/include',              # <--- 2. Homebrew General
        f'{prefix}/opt/libomp/include'    # <--- 3. OpenMP Specific
    ]
    
    library_dirs = [f'{prefix}/lib', f'{prefix}/opt/libomp/lib']
    
    c_args += ['-Xpreprocessor', '-fopenmp']
    l_args += ['-lomp']
    
else:
    # Linux / Windows
    include_dirs = [numpy.get_include()]
    library_dirs = []
    c_args += ['-fopenmp', '-march=native']
    l_args += ['-fopenmp']

# --- DEFINE EXTENSION ---
module = Extension(
    'customcv',
    sources=['customcvmodule.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=c_args,
    extra_link_args=l_args,
    language='c++'
)

setup(
    name='customcv',
    ext_modules=[module]
)