from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="leiden_clustering",
    version="0.1.0",
    packages=["leiden_clustering"],
    python_requires=">=3.7",
    package_data={"": ["LICENSE", "*.md","*.ipynb","*.yml"]},
    install_requires=[
        "numpy",
        "scanpy",
        "scikit-learn",
        "umap",
        "leidenalg"
    ],
    author="Miquel Anglada Girotto",
    author_email="miquelangladagirotto@gmail.com",
    description="Cluster your data matrix with the Leiden algorithm.",
    long_description=readme,
    url="https://github.com/MiqG/leiden_clustering",
    project_urls={"Issues": "https://github.com/MiqG/leiden_clustering/issues"},
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Topic :: System :: Clustering"
    ],
    license="BSD 3-Clause License"
)
