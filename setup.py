from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

print(requirements)

setup(
    name='GizaAutoML',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    license='MIT',
)
