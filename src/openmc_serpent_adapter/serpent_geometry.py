# SPDX-FileCopyrightText: 2023-2024 UChicago Argonne, LLC
# SPDX-License-Identifier: MIT

from math import sqrt

import openmc
from openmc.model.surface_composite import CompositeSurface


class RectangularPrism(CompositeSurface):
    """Rectangular prism as a composite surface"""

    _surface_names = ('xmin', 'xmax', 'ymin', 'ymax')

    def __init__(self, xmin, xmax, ymin, ymax, **kwargs):
        if xmin >= xmax:
            raise ValueError('xmin must be less than xmax')
        if ymin >= ymax:
            raise ValueError('ymin must be less than ymax')
        self.xmin = openmc.XPlane(x0=xmin, **kwargs)
        self.xmax = openmc.XPlane(x0=xmax, **kwargs)
        self.ymin = openmc.YPlane(y0=ymin, **kwargs)
        self.ymax = openmc.YPlane(y0=ymax, **kwargs)

    def __neg__(self):
        return -self.xmax & +self.xmin & -self.ymax & +self.ymin

    def __pos__(self):
        return +self.xmax | -self.xmin | +self.ymax | -self.ymin


def sqc(x0, y0, d, **kwargs):
    """Infinite square prism parallel to z-axis"""
    return RectangularPrism(x0 - d, x0 + d, y0 - d, y0 + d, **kwargs)


class Y_typeHexagonalPrism(CompositeSurface):
    """Y-type hexagonal prism as a composite surface"""

    _surface_names = ('right', 'left', 'upper_right', 'upper_left', 'lower_right', 'lower_left')

    def __init__(self, right, left, upper_right, upper_left, lower_right, lower_left, **kwargs):
        self.right = openmc.XPlane(x0=right, **kwargs)
        self.left = openmc.XPlane(x0=left, **kwargs)
        c = sqrt(3.)/3
        self.upper_right = openmc.Plane(a=c, b=1., d=upper_right, **kwargs)
        self.upper_left = openmc.Plane(a=-c, b=1., d=upper_left, **kwargs)
        self.lower_right = openmc.Plane(a=-c, b=1., d=lower_right, **kwargs)
        self.lower_left = openmc.Plane(a=c, b=1., d=lower_left, **kwargs)

    @CompositeSurface.boundary_type.setter
    def boundary_type(self, boundary_type):
        if boundary_type == 'periodic':
            # set the periodic BC on pairs of surfaces
            self.right.periodic_surface = self.left
            self.upper_right.periodic_surface = self.lower_left
            self.upper_left.periodic_surface = self.lower_right
        for name in self._surface_names:
            getattr(self, name).boundary_type = boundary_type

    def __neg__(self):
        return -self.right & +self.left & -self.upper_right & -self.upper_left & +self.lower_right & +self.lower_left

    def __pos__(self):
        return +self.right | -self.left | +self.upper_right | +self.upper_left | -self.lower_right | -self.lower_left


def hexyc(x0, y0, d, **kwargs):
    """Infinite hexagonal prism parallel to z-axis, flat surface perpendicular to y-axis"""
    c = sqrt(3.)/3
    l = d / (sqrt(3.)/2)
    return Y_typeHexagonalPrism(x0 + d, x0 - d, l+x0*c+y0, l-x0*c+y0, -l-x0*c+y0, -l+x0*c+y0, **kwargs)


class X_typeHexagonalPrism(CompositeSurface):
    """X-type hexagonal prism as a composite surface"""

    _surface_names = ('top', 'bottom', 'upper_right', 'upper_left', 'lower_right', 'lower_left')

    def __init__(self, top, bottom, upper_right, upper_left, lower_right, lower_left, **kwargs):
        self.top = openmc.YPlane(y0=top, **kwargs)
        self.bottom = openmc.YPlane(y0=bottom, **kwargs)
        c = sqrt(3.)/3
        self.upper_right = openmc.Plane(a=c, b=1., d=upper_right, **kwargs)
        self.upper_left = openmc.Plane(a=-c, b=1., d=upper_left, **kwargs)
        self.lower_right = openmc.Plane(a=-c, b=1., d=lower_right, **kwargs)
        self.lower_left = openmc.Plane(a=c, b=1., d=lower_left, **kwargs)

    @CompositeSurface.boundary_type.setter
    def boundary_type(self, boundary_type):
        if boundary_type == 'periodic':
            # set the periodic BC on pairs of surfaces
            self.right.periodic_surface = self.left
            self.upper_right.periodic_surface = self.lower_left
            self.upper_left.periodic_surface = self.lower_right
        for name in self._surface_names:
            getattr(self, name).boundary_type = boundary_type

    def __neg__(self):
        return -self.top & +self.bottom & -self.upper_right & -self.upper_left & +self.lower_right & +self.lower_left

    def __pos__(self):
        return +self.top | -self.bottom | +self.upper_right | +self.upper_left | -self.lower_right | -self.lower_left


def hexxc(x0, y0, d, **kwargs):
    """Infinite hexagonal prism parallel to z-axis, flat surface perpendicular to x-axis"""
    c = sqrt(3.)/3
    l = d / (sqrt(3.)/2)
    return X_typeHexagonalPrism(x0 + d, x0 - d, c*l+x0*c+y0, c*l-x0*c+y0, -c*l-x0*c+y0, -c*l+x0*c+y0, **kwargs)


def vertical_stack(z_values, universes, x0=0.0, y0=0.0, universe_id=None):
    """Vertical stack that mimics Serpent lattice type 9

    Parameters
    ----------
    z_values : iterable of float
        List of ascending z values in [cm]
    universes : iterable of openmc.Universe
        Universes to fill between each pair of z-values. The space below the
        first z value is assumed to be empty, and the last universe fills all
        space above the last z value.
    x0, y0 : float
        x- and y-coordinates of the vertical stack origin in [cm]

    Returns
    -------
    openmc.Universe
        Vertical stack

    """
    # Create z-planes and regions between them
    z_planes = [openmc.ZPlane(z) for z in z_values]
    regions = openmc.model.subdivide(z_planes)

    # Create cells for each region
    cells = [openmc.Cell(region=r) for r in regions]

    # Fill cells with universes and set translation
    for cell, univ in zip(cells[1:], universes):
        cell.fill = univ
        cell.translation = (x0, y0, 0.0)

    # Create universe with vertical stack
    return openmc.Universe(universe_id=universe_id, cells=cells)

class Z_Vessel(CompositeSurface):
    """Vessel as a composite surface parallel to z-axis"""

    _surface_names = ('cycl', 'zmin', 'zmax', 'bottom', 'top')

    def __init__(self, x0, y0, r, zmin, zmax, hbottom, htop, **kwargs):
        if zmin >= zmax:
            raise ValueError('zmin must be less than zmax')
        
        self.cycl = openmc.ZCylinder(x0=x0, y0=y0, r=r, **kwargs)
        self.zmin = openmc.ZPlane(z0=zmin, **kwargs)
        self.zmax = openmc.ZPlane(z0=zmax, **kwargs)

        """
        Coefficients for quadric surface to create an ellipsoid

            General equation for an ellipsoid: 
                (x-xo)^2/r^2 + (y-yo)^2/r^2 + (z-zo)^2/h^2 = 1
                
            General form of a quadric surface equation:
                Ax^2 + By^2 + Cz^2 + Gx + Hy + Jz + K = 0
                """

        A = 1/r**2
        B = 1/r**2
        C1 = 1/hbottom**2
        C2 = 1/htop**2
        G = -(2*x0)/r**2
        H = -(2*y0)/r**2
        J1 = -(2*zmin)/hbottom**2
        J2 = -(2*zmax)/htop**2
        K1 = x0**2/r**2 + y0**2/r**2 + zmin**2/hbottom**2 - 1
        K2 = x0**2/r**2 + y0**2/r**2 + zmax**2/htop**2 - 1

        self.bottom = openmc.Quadric(a=A, b=B, c=C1, g=G, h=H, j=J1, k=K1, **kwargs)
        self.top = openmc.Quadric(a=A, b=B, c=C2, g=G, h=H, j=J2, k=K2, **kwargs)
    
    def __neg__(self):
        return (-self.cycl & +self.zmin & -self.zmax) | (-self.bottom & -self.zmin) | (-self.top & +self.zmax)
    
    def __pos__(self):
        return (+self.cycl | -self.zmin | +self.zmax) & (+self.bottom | +self.zmin) & (+self.top | -self.zmax)

def zvessel(x0, y0, r, zmin, zmax, hbottom, htop, **kwargs):
    """Vessel parallel to z-axis"""
    return Z_Vessel(x0, y0, r, zmin, zmax, hbottom, htop, **kwargs)
