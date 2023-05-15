#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test all the example Jupyter notebooks.
For now we just check that they run without errors.
"""
import sys
import os
import pytest
import inspect
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nbpath = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "..", "examples")

# All notebooks we want to test in the list below.
# Note that the synth example and multi layer icequake example are
# is very slow (doing the direct two layer cell) so are excluded
notebooks_to_test = ["paper_examples/sks_figure/sks_figure.ipynb",
                     #"paper_examples/synth_example_figures/synth_example.ipynb", 
                     #"multi_layer_icequake_figure/multi_layer_icequake_figure.ipynb",
                     "automated_example/automated_example.ipynb",
                     "paper_examples/icequake_figure/icequake_figure.ipynb"]

def _run_notebook(nb_path):
    """Helper to run a notebook and return any errors."""

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(allow_errors=True, kernel_name='python3',
                                 timeout=1200)
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})

        # Check for any errors and return a list of error cells
        err = [out for cell in nb.cells if "outputs" in cell
                   for out in cell["outputs"]\
                   if out.output_type == "error"]

    return err


@pytest.mark.parametrize("notebook", notebooks_to_test)
def test_notebook(notebook):
    err = _run_notebook(os.path.join(nbpath, notebook))
    assert err == []

