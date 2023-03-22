Coordinate System
=================

It is important to understand the coordinate system used by swspy, given that swspy can measure ansiotropy in 3D, as opposed to most previous shear-wave splitting codes.

There are two options that one can use for coordinate system:
1. ZNE - In this coordinate system, the package assumes that the ray is coming in at vertical incidence, therefore measuring fst-direction in the horizontal plane. This is conventionally how most previous codes have performed shear-wave splitting analysis.
2. LQT - In this coordinate system, the package works in the emerging coordinate system, i.e. in 3D. In this system, the fast direction is described by two angles, phi_1 and phi_2.

Below is a figure summarising the coordinate system. In both the above cases, the problem is solved in the QT plane then translated to the correct coordinate system for the outputs. For the ZNE coordinate system option, theta_inc = 0.


.. image:: images/coordinate_systems_figure.png
  :width: 600