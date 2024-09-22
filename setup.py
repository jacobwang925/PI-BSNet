from setuptools import setup, find_packages

setup(
   name='bs_net',
   version='0.0.1',
   description='bspline_net',
   author='Jacob Wang',
   author_email='',
   package_dir={"": "src"},
   packages=find_packages(where="src", include=["src*"]), 
   install_requires=['matplotlib', 'numpy', 'scipy','fire'], #external packages as dependencies
)