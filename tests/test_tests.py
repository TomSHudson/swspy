# Test cases for testing
# 
# We have some testing helpers. These tests test that the testing helpers
# work. It's good if they do. Or we know if they don't.

import pytest
import numpy as np
import swspy


def test_clamp_angle():

    ins = np.array((90.0, -89.9, 91.0, -90.0, -91.0, 0.0, 359.0, 360.0, 1.0))
    outs = np.array((90.0, -89.9, -89.0, 90.0, 89.0, 0.0, -1.0, 0.0, 1.0)) 

    calc_outs = swspy.testing.clamp_angle(ins)
    np.testing.assert_allclose(outs, calc_outs)
     

def test_periodic_angular_difference():

    a_in = np.array([359, 89, 179, 271, 0.0, 90.0, -90.0, 360.0])
    b_in = np.array([1, 91, 181, 269, 180.0, -90.0, 90.0, 180.0])
    diff_in = np.array([2., 2., 2., 2., 0., 0., 0., 0.])
    diff = swspy.testing.periodic_angular_difference(a_in, b_in)
    np.testing.assert_allclose(diff_in, diff)


def test_assert_angle_allclose_close():

    a_in = np.array([359, 89, 179, 271, 0.0, 90.0, -90.0, 360.0])
    b_in = np.array([1, 91, 181, 269, 180.0, -90.0, 90.0, 180.0])
    swspy.testing.assert_angle_allclose(a_in, b_in, atol=2.1, rtol=0.0)


def test_assert_angle_allclose_raises():
    a_in = np.array([0.0, 45.0, 90.0])
    b_in = np.array([30.0, 30.0, 30.0])
    with pytest.raises(AssertionError):
        swspy.testing.assert_angle_allclose(a_in, b_in)
