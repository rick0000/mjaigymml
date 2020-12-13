from setuptools import setup, find_packages

setup(
    name="mjaigym_ml",
    version="0.2.0",
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
