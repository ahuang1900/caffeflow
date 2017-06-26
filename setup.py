#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import pypandoc

setup(
    name="caffeflow",
    version="0.1.1",
    description="Utility to convert Caffe models to TensorFlow.",
    long_description=pypandoc.convert('README.md', 'rst'),
    url="https://github.com/xmartlabs/caffeflow",
    keywords=["Caffe", "machine learning", "artificial intelligence", "model", "utility", "TensorFlow"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'caffeflow = caffeflow.convert:main',
        ],
    },
    install_requires=[
        'tensorflow>=1.2.0',
    ],
    author="Xmartlabs",
    author_email="hi@xmartlabs.com",
    maintainer="Santiago Castro",
    maintainer_email="santiago@xmartlabs.com",
    license="MIT",
)
