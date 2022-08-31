from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Unplanned downtime of wind turbines can result in a significant loss of revenue and energy and can easily scale to millions of dollars a year. This project is to create a model to get an ideally functioning turbine’s expected rotor bearing temperature. It will then use the model to check the deviation of the actual rotor bearing temperature of the faulty turbine from the expected temperature. ',
    author='Aditya Jha',
    license='MIT',
)
