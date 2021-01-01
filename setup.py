from setuptools import setup, find_packages

setup(
    name="mjaigymml",
    version="0.4.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "joblib",
        "dataclasses",
        "mlflow",
        "pyyaml",
        "mjaigym @ git+https://github.com/rick0000/mjaigym.git@0.2.0",
        "h5py",
    ],
)
