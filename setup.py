from setuptools import setup, find_packages

setup(
   name='bsnet',
   description='bspline_net',
   author='Jasmine Ratchford',
   author_email='jratchford@sei.cmu.edu',
   version='0.1.0',
   package_dir={'': 'src'},
   packages=find_packages(where='src'),
   install_requires=['matplotlib', 'numpy', 'scipy','fire', 'torch', 'tensorboard', 'lightning', 'opt-einsum','jsonargparse[signatures]>=4.27.7'], #external packages as dependencies
)