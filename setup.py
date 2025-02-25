from setuptools import setup, find_packages

setup(
    name="japanese_occupation_classifier",
    version="0.1",
    packages=find_packages(where="Code"),
    package_dir={"": "Code"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "fugashi>=1.1.0",
        "ipadic>=1.0.0",
        "jaconv>=0.3.0",
        "joblib>=1.1.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.62.0",
    ],
) 