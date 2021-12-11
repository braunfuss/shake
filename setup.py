#!/usr/bin/python3

from setuptools import setup, find_packages

test_requirements = ["pylint", "pytest", "pre-commit"]

setup(
    name="shake",
    version="0.1.0",
    description='Shakemaps',
    license='GPLv3',
    author_email='andreas.steinberg@bgr.de',
    extras_require={"tests": test_requirements},
    install_requires=[
        "geopandas",
        "pyyaml",
        "Rtree",
        "rasterio",
    ],
    packages=[
        'shake',
        'shake.apps',
        'shake.oq_shakemap',
        'shake.oq_shakemap.io',
        'shake.util',
        'shake.syn_shake',
        'shake.oq_shakemap.regionalization_files',
        ],
    entry_points={
        'console_scripts': [
            'shake = shake.apps.shakeit:main',
            'shakedown = shake.apps.shakedown:main',
        ]
    },
    package_dir={'shake': 'src'},
    python_requires=">=3.7",
    package_data={
        'shake': [
            'oq_shakemap/regionalization_files/*',
            ]},
)
