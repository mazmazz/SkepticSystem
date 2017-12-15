from setuptools import find_packages, setup

setup(
    name='skepticsys',
    version='0.0.1',
    description='Python component of SkepticSystem',
    url='https://github.com/mazmazz',
    author='mazmazz',
    packages=['skepticsys'],
    install_requires=[
        'numpy>=1.11.3',
        'scikit-learn>=0.18.2'
    ]
)
