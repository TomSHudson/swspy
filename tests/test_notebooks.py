#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test all the example Jupyter notebooks.
For now we just check that they run without errors.
"""
import sys
import os
import inspect
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nbpath = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "..", "examples")

def _run_notebook(nb_path):
    """Helper to run a notebook and return any errors."""

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(allow_errors=True, kernel_name='python3',
                                 timeout=600)
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})

        # Check for any errors and return a list of error cells
        err = [out for cell in nb.cells if "outputs" in cell
                   for out in cell["outputs"]\
                   if out.output_type == "error"]

    return err

def test_sks_figure():
    err = _run_notebook(os.path.join(nbpath, "paper_examples/sks_figure/sks_figure.ipynb"))
    assert err == []

