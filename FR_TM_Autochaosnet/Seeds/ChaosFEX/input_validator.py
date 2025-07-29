# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:21:03 2023

@author: Akhila Henry

Description:
------------
This module provides basic input validation utilities for checking the structure 
and properties of feature matrices used in data-driven algorithms such as 
Neurochaos Learning or other ML workflows.

Key Components:
---------------
1. _check_features(feat_mat):
    - Checks if the input `feat_mat` is a 2D NumPy array (`ndarray`) of type `float64`.
    - Verifies that all values lie within the closed interval [0, 1].
    - Returns `True` if all checks pass, otherwise prints an error message and returns `False`.

2. validate(feat_mat):
    - A wrapper function that calls `_check_features`.
    - Returns `True` if feature matrix is valid, otherwise `False`.

Usage:
------
Use this module to validate input feature matrices before applying further 
feature transformation or learning steps. Ensures data consistency and range safety.

Dependencies:
-------------
- numpy
"""

# Import calls
from numpy import ndarray, float64


def _check_features(feat_mat):

    # Check type and shape of input feature matrix
    if not (
        isinstance(feat_mat, ndarray)
        and feat_mat.dtype == float64
        and feat_mat.ndim == 2
    ):
        print("> ERROR: feat_mat should be 2D array of dtype float64 ...")
        return False

    # Check ranges of values in input feature matrix
    if feat_mat.min() < 0 or feat_mat.max() > 1:
        print("> ERROR: feat_mat should be scaled between 0 & 1 ...")
        return False

    return True




def validate(feat_mat):

    if (_check_features(feat_mat)):
        return True
    else:
        return False
