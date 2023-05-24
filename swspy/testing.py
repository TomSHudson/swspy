# Helper functions for testing swspy

import numpy as np

def clamp_angle(angle_in):
    """
    Assuming 180 degree periodicity, represent an angle between -90 and +90 degrees

    angle is a numpy array.
    """
    angle = np.atleast_1d(angle_in)
    angle = angle - 180.0 * np.fix(angle / 180.0)
    angle[angle<=-90.0] = angle[angle<=-90.0] + 180.0
    angle[angle>90.0] = angle[angle>90.0] - 180.0

    return angle


def periodic_angular_difference(a_angle, b_angle):
    """
    Angular difference assuming 180 degree periodicity
    
    a_angle and b_angle are numpy arrays of angle, retuns a numpy array of
    differences such that 0.0 and 360.0 and 180.0 are all the same.
    """
    difference = np.minimum(np.fabs(a_angle - b_angle), 
                       np.fabs(clamp_angle(a_angle) - clamp_angle(b_angle)))
    return difference


def assert_angle_allclose(actual, desired, **kwargs):
    """
    Raises an AssertionError if two objects are not equal accounting for angle

    This wraps numpy.testing.assert_allclose but values such as 0.0, 360.0 and
    180.0 are all treated as being the same. Keyword arguments are passed
    on.
    """
    diff = periodic_angular_difference(actual, desired)
    real_desired = np.zeros_like(desired)
    np.testing.assert_allclose(diff, real_desired, **kwargs)
