#---------------------------------------------------------------------
# Input conversion code from SERPENT to OpenMC
#---------------------------------------------------------------------
import argparse

import openmc
import matplotlib.pyplot as plt
import numpy as np

from openmc.model.surface_composite import CompositeSurface
from openmc.data import get_thermal_name
from openmc.data.ace import get_metadata
from math import sqrt

from math import log10

#---------------------------------------------------------------------
# Defining the rectangular prism as a composite surface
class RectangularPrism(CompositeSurface):
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
    return RectangularPrism(x0 - d, x0 + d, y0 - d, y0 + d, **kwargs)

#---------------------------------------------------------------------
# Defining the y-type hexagonal prism as a composite surface
class Y_typeHexagonalPrism(CompositeSurface):
    _surface_names = ('right', 'left', 'upper_right', 'upper_left', 'lower_right', 'lower_left')

    def __init__(self, right, left, upper_right, upper_left, lower_right, lower_left, **kwargs):
        self.right = openmc.XPlane(x0=right, **kwargs)
        self.left = openmc.XPlane(x0=left, **kwargs)
        c = sqrt(3.)/3
        self.upper_right = openmc.Plane(a=c, b=1., d=upper_right, **kwargs)
        self.upper_left = openmc.Plane(a=-c, b=1., d=upper_left, **kwargs)
        self.lower_right = openmc.Plane(a=-c, b=1., d=lower_right, **kwargs)
        self.lower_left = openmc.Plane(a=c, b=1., d=lower_left, **kwargs)

    def __neg__(self):
        return -self.right & +self.left & -self.upper_right & -self.upper_left & +self.lower_right & +self.lower_left
    def __pos__(self):
        return +self.right | -self.left | +self.upper_right | +self.upper_left | -self.lower_right | -self.lower_left
def hexyc(x0, y0, d, **kwargs):
    c = sqrt(3.)/3
    l = d / (sqrt(3.)/2)
    return Y_typeHexagonalPrism(x0 + d, x0 - d, l+x0*c+y0, l-x0*c+y0, -l-x0*c+y0, -l+x0*c+y0, **kwargs)

#---------------------------------------------------------------------
# Defining the x-type hexagonal prism as a composite surface
class X_typeHexagonalPrism(CompositeSurface):
    _surface_names = ('top', 'bottom', 'upper_right', 'upper_left', 'lower_right', 'lower_left')

    def __init__(self, top, bottom, upper_right, upper_left, lower_right, lower_left, **kwargs):
        self.top = openmc.YPlane(y0=top, **kwargs)
        self.bottom = openmc.YPlane(y0=bottom, **kwargs)
        c = sqrt(3.)/3
        self.upper_right = openmc.Plane(a=c, b=1., d=upper_right, **kwargs)
        self.upper_left = openmc.Plane(a=-c, b=1., d=upper_left, **kwargs)
        self.lower_right = openmc.Plane(a=-c, b=1., d=lower_right, **kwargs)
        self.lower_left = openmc.Plane(a=c, b=1., d=lower_left, **kwargs)

    def __neg__(self):
        return -self.top & +self.bottom & -self.upper_right & -self.upper_left & +self.lower_right & +self.lower_left
    def __pos__(self):
        return +self.top | -self.bottom | +self.upper_right | +self.upper_left | -self.lower_right | -self.lower_left
def hexyc(x0, y0, d, **kwargs):
    c = sqrt(3.)/3
    l = d / (sqrt(3.)/2)
    return X_typeHexagonalPrism(x0 + d, x0 - d, c*l+x0*c+y0, c*l-x0*c+y0, -c*l-x0*c+y0, -c*l+x0*c+y0, **kwargs)

#---------------------------------------------------------------------
# Definig a new surface ID with all integers
def update_openmc_surfaces(openmc_surfaces):
    # Determine maximum integer ID
    max_id = 0
    for surface_id in openmc_surfaces.keys():
        if surface_id.isnumeric():
            max_id = max(max_id, int(surface_id))

    # Change non-numeric keys to numeric
    strid_to_intid = {}
    for surface_id, surface in openmc_surfaces.items():
        if not surface_id.isnumeric():
            max_id += 1
            strid_to_intid[surface_id] = max_id

    for str_id, int_id in strid_to_intid.items():
        openmc_surfaces[str(int_id)] = openmc_surfaces[str_id]
        del openmc_surfaces[str_id]

    return strid_to_intid
#----------------------------------------------------------------------
# For the conversion of vertical stack lattice (9)
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
#----------------------------------------------------------------------
openmc_surfaces  = {}
openmc_cells     = {}
openmc_universes = {}
therm_materials  = {}
openmc_materials = {}
openmc_lattices  = {}
therm_materials  = {}

parser = argparse.ArgumentParser()
parser.add_argument('input_file', nargs='?', default='main')
args = parser.parse_args()

# Combine all files into one big list
all_lines = []
with open(args.input_file, 'r') as file_handle:
    for line in file_handle:
        words = line.split()
        if words and words[0] == 'include':
            filename = words[1][1:-1]
            all_lines.extend(open(filename, 'r').readlines())
        else:
            all_lines.append(line)
#-----------------------------------------------------------------------------
# Conversion of a SERPENT material to an OpenMC material
# Thermal scattering materials
for line in all_lines:
    words = line.split()
    # Get rid of comments
    for word in words:
        if word[0] == '%':
            try:
                index_comment = words.index(word)
                words = words[:index_comment]
            except ValueError:
                # If no comment, do nothing
                pass
    if len(words) > 3 and words[0] == 'therm':
        mat_id                              = words[1]
        mat_temp                            = words[2]
        materials                           = [words[3]]

        if '.' in words[3]:
            name, xs = words[3].split('.')
        else:
            name = words[3]
        therm_materials[mat_id] = get_thermal_name(str(name))
    elif len(words) > 0 and len(words) <= 3 and words[0] == 'therm':
        mat_id                              = words[1]
        materials                           = [words[2]]

        if '.' in words[2]:
            name, xs = words[2].split('.')
        else:
            name = words[2]
        therm_materials[mat_id] = get_thermal_name(str(name))
# Materials
for line in all_lines:
    words = line.split()
    # Get rid of comments
    for word in words:
        if word[0] == '%':
            try:
                index_comment = words.index(word)
                words = words[:index_comment]
            except ValueError:
                # If no comment, do nothing
                pass

    if len(words) > 0 and words[0] == 'mat':
        control = 'mat'
    elif len(words) > 0 and (words[0] == 'lat' or words[0] == 'surf' or words[0] == 'cell' or words[0] == 'mix' or words[0] == 'set'):
        control = 0

    # Material cards
    if len(words) > 0 and words[0] == 'mat':
        mat_id                      = words[1]
        openmc_materials[mat_id]    = openmc.Material()
        if words[2] != 'sum':
            if words[2][0] == '-':
                words[2] = words[2][1:]
                openmc_materials[mat_id].set_density('g/cm3', float(words[2]))
            else:
                openmc_materials[mat_id].set_density('atom/b-cm', float(words[2]))
        for x in range(len(words)):
            if words[x] == 'tmp':
                mat_temp                             = float(words[x+1])
                openmc_materials[mat_id].temperature = float(mat_temp)
            # OpenMC does not support thermal scattering tables in mixing materials
            # For the future and the general use of conversion file, it needs to be fixed in the source code!!!!
            elif words[x] == 'moder':
                    openmc_materials[mat_id].add_s_alpha_beta(str(therm_materials[str(words[x+1])]))
        mix     = []
        mix_per = []
    # Mixture cards
    elif len(words) > 0 and words[0] == 'mix':
        mat_id                      = words[1]
        openmc_materials[mat_id]    = openmc.Material()
        mix     = []
        mix_per = []

    elif len(words) > 0 and words[0] in openmc_materials:
        mix_id      = words[0]
        mix_percent = float(words[1])/100
        percent     = f'{mix_percent:.3f}'
        mix.append(openmc_materials[mix_id])
        mix_per.append(float(percent))
        if mix_percent > 0:
            openmc_materials[mat_id]    = openmc.Material.mix_materials(mix, mix_per, 'vo')
        else:
            openmc_materials[mat_id]    = openmc.Material.mix_materials(mix, mix_per, 'wo')
    # Adding materials to the material card
    elif len(words) > 0 and words[0] != 'mix' and words[0] != 'mat' and words[0] != 'therm' and words[0] != 'cell' and words[0] != 'surf' and words[0] != 'set' and words[0] != 'lat' and words[0] not in openmc_materials and control == 'mat':
        nuclide = words[0]
        percent = float(words[1])
        if '.' in nuclide:
            zaid, xs = nuclide.split('.')
        else:
            zaid = nuclide
        name, element, Z, A, metastable = get_metadata(int(zaid))
        if percent < 0:
            if A > 0:
                openmc_materials[mat_id].add_nuclide(name, abs(percent), 'wo')
            else:
                openmc_materials[mat_id].add_element(element, abs(percent), 'wo')
        else:
            if A > 0:
                openmc_materials[mat_id].add_nuclide(name, abs(percent), 'ao')
            else:
                openmc_materials[mat_id].add_element(element, abs(percent), 'ao')

#-----------------------------------------------------------------------------
# Conversion of a SERPENT surface to an OpenMC surface
for line in all_lines:
    words = line.split()
    if len(words) > 0 and words[0] == 'surf':
        # Get rid of comments
        for word in words:
            if word[0] == '%':
                try:
                    index_comment = words.index(word)
                    words = words[:index_comment]
                except ValueError:
                    # If no comment, do nothing
                    pass
    if len(words) > 0 and words[0] == 'set' and words[1] == 'bc':
        boundary = words[2]
    if len(words) > 0 and words[0] == 'surf':

        # Read ID, surface type and coefficients
        surface_id = words[1]
        surface_type = words[2]
        coefficients = [words[3:]]
        coefficients = [float(x) for x in words[3:]]

        # Convert to OpenMC surface and add to dictionary
        if surface_type == 'pz':
            openmc_surfaces[words[1]] = openmc.ZPlane(coefficients[0])
        elif surface_type == 'px':
            openmc_surfaces[words[1]] = openmc.XPlane(coefficients[0])
        elif surface_type == 'py':
            openmc_surfaces[words[1]] = openmc.YPlane(coefficients[0])
        elif surface_type in ('cyl', 'cylz'):
            if len(coefficients) == 3:
                x0, y0, r = coefficients[0], coefficients[1], coefficients[2]
                openmc_surfaces[words[1]] = openmc.ZCylinder(x0, y0, r)
            elif len(coefficients) == 5:
                x0, y0, r, z0, z1 = coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
                center_base = x0, y0, z0
                height = z1 - z0
                radius = r
                openmc_surfaces[words[1]] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='z')
        elif surface_type == ('cylx'):
            if len(coefficients) == 3:
                y0, z0, r = coefficients[0], coefficients[1], coefficients[2]
                openmc_surfaces[words[1]] = openmc.XCylinder(y0, z0, r)
            elif len(coefficients) == 5:
                y0, z0, r, x0, x1 = coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
                center_base = x0, y0, z0
                height = x1 - x0
                radius = r
                openmc_surfaces[words[1]] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='x')
        elif surface_type == ('cyly'):
            if len(coefficients) == 3:
                x0, z0, r = coefficients[0], coefficients[1], coefficients[2]
                openmc_surfaces[words[1]] = openmc.YCylinder(x0, z0, r)
            elif len(coefficients) == 5:
                x0, z0, r, y0, y1 = coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
                center_base = x0, y0, z0
                height = y1 - y0
                radius = r
                openmc_surfaces[words[1]] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='y')
        elif surface_type == ('sqc'):
            x0, y0, half_width = coefficients[0], coefficients[1], coefficients[2]
            openmc_surfaces[words[1]] = sqc(x0, y0, half_width)
        elif surface_type == ('torx'):
            x0, y0, z0, A, B, C = coefficients
            openmc_surfaces[words[1]] = openmc.XTorus(x0, y0, z0, A, B, C)
        elif surface_type == ('tory'):
            x0, y0, z0, A, B, C = coefficients
            openmc_surfaces[words[1]] = openmc.YTorus(x0, y0, z0, A, B, C)
        elif surface_type == ('torz'):
            x0, y0, z0, A, B, C = coefficients
            openmc_surfaces[words[1]] = openmc.ZTorus(x0, y0, z0, A, B, C)
        elif surface_type == ('sph'):
            x0, y0, z0, r = coefficients
            openmc_surfaces[words[1]] = openmc.Sphere(x0, y0, z0, r)
        elif surface_type == ('plane'):
            A, B, C, D = coefficients
            openmc_surfaces[words[1]] = openmc.Plane(A, B, C, D)
        elif surface_type == ('cone'):
            x0, y0, z0, r, h = coefficients
            R = coefficients[3]/coefficients[4]
            Z0 = coefficients[4]+coefficients[2]
            up = (h < 0)
            openmc_surfaces[words[1]] = openmc.model.ZConeOneSided(x0, y0, Z0, R, up)
        elif surface_type == ('hexyc'):
            x0, y0, d = coefficients[0], coefficients[1], coefficients[2]
            openmc_surfaces[words[1]] = hexyc(x0, y0, d)
# Conversion of string surface ids to integers ids
name_to_id = update_openmc_surfaces(openmc_surfaces)
keys = list(openmc_surfaces.keys())
for key in keys:
    openmc_surfaces[int(key)] = openmc_surfaces[key]
    del openmc_surfaces[key]

#--------------------------------------------------------------------------------
#Conversion of a SERPENT cell and universe to a OpenMC cell and universe
outer_surfaces = []
inner_surfaces = []
for line in all_lines:
    words = line.split()
    if len(words) > 0 and words[0] == 'cell':
    # Get rid of comments
        try:
            index_comment = words.index('%')
            words         = words[:index_comment]
        except ValueError:
            # If no comment, do nothing
            pass

        # Read ID, universe, material and coefficients
        cell_id = words[1]
        cell_universe = words[2]
        if cell_universe not in openmc_universes:
            openmc_universes[cell_universe]  = openmc.Universe()
        if words[3] == 'fill':
            coefficients            = [str(x) for x in words[5:]]
            openmc_cells[cell_id]        = openmc.Cell()
        elif words[3] == 'void':
            openmc_cells[cell_id]   = openmc.Cell(fill=None)
            coefficients            = [str(x) for x in words[4:]]
        elif words[3] == 'outside':
            openmc_cells[cell_id]   = openmc.Cell(fill=None)
            coefficients            = [str(x) for x in words[4:]]
        else:
            cell_material           = words[3]
            cell_material           = openmc_materials[cell_material]
            coefficients            = [str(x) for x in words[4:]]
        for x in range(len(coefficients)-1, 0, -1):
            if coefficients[x]=='-':
                coefficients[x+1]=(f'-{coefficients[x+1]}')
                del(coefficients[x])

        # Creating regions
        coefficient = ' '.join(coefficients)
        for name, surface_id in sorted(name_to_id.items(), key=lambda x: len(x[0]), reverse=True):
            coefficient = coefficient.replace(name, str(surface_id))
        try:
            cell_region  = openmc.Region.from_expression(expression = coefficient, surfaces = openmc_surfaces)
        except Exception:
            print(f'Failed on line: {line}')
            raise

        # Outer boundary conditions
        for x in range(len(coefficients)):
            for name, surface_id in sorted(name_to_id.items(), key=lambda x: len(x[0]), reverse=True):
                coefficients[x] = coefficients[x].replace(name, str(surface_id))
        if words[3] != 'outside':
            for coefficient in coefficients:
                if coefficient not in inner_surfaces:
                    inner_surfaces.append(coefficient)
        if words[3] == 'outside':
            #print(f'cell_id = {words[1]}, coefficients = {coefficients}')
            for coefficient in coefficients:
                if coefficient not in outer_surfaces:
                    outer_surfaces.append(coefficient)
            for surface in outer_surfaces:
                for name in inner_surfaces:
                    if surface == name :
                        outer_surfaces.remove(surface)

        # Convert to OpenMC cell and add to dictionary
        if words[3] == 'fill':
            if words[4] in openmc_universes:
                openmc_cells[cell_id]        = openmc.Cell(fill=openmc_universes[words[4]], region=cell_region)
            elif words[4] in openmc_lattices:
                openmc_cells[cell_id]        = openmc.Cell(fill=openmc_lattices[words[4]], region=cell_region)
        elif words[3] == 'void':
            openmc_cells[cell_id]            = openmc.Cell(fill=None, region=cell_region)
        elif words[3] == 'outside':
            openmc_cells[cell_id]            = openmc.Cell(fill=None, region=cell_region)
        else:
            openmc_cells[cell_id]            = openmc.Cell(fill=cell_material, region=cell_region)

        # Adding the cell to a universe
        if words[3] == 'fill':
            continue
        else:
            openmc_universes[cell_universe].add_cell(openmc_cells[cell_id])
for surface in outer_surfaces:
    if '-' in surface:
        surface = surface[1:]
    if boundary == '1':
        boundary = 'vacuum'
    elif boundary == '2':
        boundary = 'reflective'
    elif boundary == '3':
        boundary = 'periodic'
    openmc_surfaces[int(surface)].boundary_type = boundary

#--------------------Pin definition---------------------------------------
ctrl = 'ctrl'
for line in all_lines:
    words = line.split()

    # Get rid of comments
    try:
        index_comment = words.index('%')
        words         = words[:index_comment]
    except ValueError:
        # If no comment, do nothing
        pass

        # Read ID, universe, material and coefficients
    if len(words) > 0 and words[0] == 'pin':
        cell_universe = words[1]
        ctrl = words[0]
        surfaces = []
        items = []
    if len(words) > 0 and ctrl == 'pin' and (words[0] == 'surf' or words[0] == 'cell' or words[0] == 'mat'or words[0] == 'lat' or words[0] == 'set' or words[0] == 'include' or words[0] == 'plot' or words[0] == 'therm' or words[0] == 'dep'):
        openmc_universes[cell_universe] = openmc.model.pin(surfaces, items, subdivisions=None, divide_vols=True)
    if len(words) > 0 and (words[0] == 'surf' or words[0] == 'cell' or words[0] == 'mat'or words[0] == 'lat' or words[0] == 'set' or words[0] == 'include' or words[0] == 'plot' or words[0] == 'therm' or words[0] == 'dep' or words[0] == 'pin'):
        ctrl = words[0]

    if len(words) > 1 and ctrl == 'pin' and words[0] != 'pin':
        material_region = words[0]
        outer_radi = float(words[1])
        surface = openmc.ZCylinder(r = outer_radi)
        surfaces.append(surface)
        item = openmc_materials[material_region]
        items.append(item)
    elif len(words) == 1 and ctrl == 'pin' and words[0] != 'pin':
        material_region = words[0]
        item = openmc_materials[material_region]
        items.append(item)
#openmc_universes[cell_universe] = openmc.model.pin(surfaces, items, subdivisions=None, divide_vols=True)
# NOTE: If there is more than one pin, this code does not work. Nedds to be fixed
#---------------------------------------------------------------------------------------------------------------------------
#Conversion of a SERPENT lattice to a OpenMC lattice
lattice_type = ''
for line in all_lines:
    words = line.split()
    # Get rid of comments
    for word in words:
        if word[0] == '%':
            try:
                index_comment = words.index(word)
                words         = words[:index_comment]
            except ValueError:
                # If no comment, do nothing
                pass

    if len(words) > 0 and words[0] == 'lat':
        z0 = []
        uni = []
        lattice_id   = words[1]
        lattice_type = words[2]
        if lattice_type == '1':
            openmc_lattices[lattice_id]             = openmc.RectLattice()
            x0, y0                                  = words[3:5]
            nx, ny                                  = words[5:7]
            pitch                                   = float(words[7])
            openmc_lattices[lattice_id].lower_left  = (-(float(nx)/2)*pitch, -(float(ny)/2)*pitch)
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '2':
            openmc_lattices[lattice_id]             = openmc.HexLattice()
            openmc_lattices[lattice_id].orientation = 'x'
            x0, y0                                  = words[3:5]
            nx, ny                                  = words[5:7]
            pitch                                   = words[7]
            openmc_lattices[lattice_id].lower_left  = (-(nx/2)*pitch, -(ny/2)*pitch)
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '3':
            openmc_lattices[lattice_id]             = openmc.HexLattice()
            openmc_lattices[lattice_id].orientation = 'y'
            x0, y0                                  = words[3:5]
            nx, ny                                  = words[5:7]
            pitch                                   = words[7]
            openmc_lattices[lattice_id].lower_left  = (-(nx/2)*pitch, -(ny/2)*pitch)
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '6':
            openmc_lattices[lattice_id]             = openmc.RectLattice()
            x0, y0                                  = words[3:5]
            pitch                                   = words[5]
            openmc_lattices[lattice_id].lower_left  = (-(x0+(pitch/2)), -(y0+(pitch/2)))
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '7':
            openmc_lattices[lattice_id]             = openmc.HexLattice()
            openmc_lattices[lattice_id].orientation = 'x'
            x0, y0                                  = words[3:5]
            pitch                                   = words[5]
            openmc_lattices[lattice_id].lower_left  = (-(x0+(pitch/2)), -(y0+(pitch/2)))
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '8':
            openmc_lattices[lattice_id]             = openmc.HexLattice()
            openmc_lattices[lattice_id].orientation = 'y'
            x0, y0                                  = words[3:5]
            pitch                                   = words[5]
            openmc_lattices[lattice_id].lower_left  = (-(x0+(pitch/2)), -(y0+(pitch/2)))
            openmc_lattices[lattice_id].pitch       = (pitch, pitch)
        elif lattice_type == '9':
            x0, y0                                  = words[3:5]
            x0, y0                                  = float(x0), float(y0)
            n                                       = float(words[5])
        # Missing circular cluster array (4), 3D cuboidal lattice (11), x-type triangular lattice (14)
        # 3D x-type hexagonal prism lattice (12), and 3D y-type hexagonal prism lattice (13)
        elif lattice_type == '4':
            print('Lattice geometry: circular cluster array is not supported!')
            quit
        elif lattice_type == '11':
            print('Lattice geometry: 3D cuboidal lattice is not supported!')
            quit
        elif lattice_type == '12':
            print('Lattice geometry: 3D x-type hexagonal prism lattice is not supported!')
            quit
        elif lattice_type == '13':
            print('Lattice geometry: 3D y-type hexagonal prism lattice is not supported!')
            quit
        elif lattice_type == '14':
            print('Lattice geometry: x-type triangular lattice is not defined supported!')
            quit

    if len(words) > 0 and (words[0] == 'surf' or words[0] == 'cell' or words[0] == 'mat'or words[0] == 'lat' or words[0] == 'set' or words[0] == 'include' or words[0] == 'plot' or words[0] == 'therm'or words[0] == 'dep'):
        ctrl = words[0]
        #print(ctrl)

    if len(words) > 0 and words[0] != 'surf' and words[0] != 'cell' and lattice_type =='9' and words[0] != 'lat' and words[0] != 'plot' and words[0] != 'set':
        z0.append(float(words[0]))
        uni.append(openmc_universes[words[1]])
        openmc_lattices[lattice_id]                 = vertical_stack(z0, uni, x0, y0)
    elif len(words) > 0 and words[0] != 'surf' and words[0] != 'mat' and ctrl != 'mat' and ctrl != 'dep' and words[0] != 'cell' and lattice_type !='9' and words[0] != 'lat'and words[0] != 'set' and words[0] != 'include' and words[0] != 'plot' and words[0] != 'therm' and words[0] != 'pin' and words[0] != 'dep' and words[0] not in openmc_materials:
        for x in range(len(words)):
            #print(words[x])
            words[x]                                = openmc_universes[words[x]]
        control                                     = words
        uni.append(control)
        lattice                                     = openmc_lattices[lattice_id]
        lattice.universes                           = list(reversed(uni))

#---------------------------------------------------------------------------------------------------------------------------
# Creating cells with 'fill' command
for line in all_lines:
    words = line.split()
    if len(words) > 0 and words[0] == 'cell':
    # Get rid of comments
        try:
            index_comment = words.index('%')
            words         = words[:index_comment]
        except ValueError:
            # If no comment, do nothing
            pass

        # Read ID, universe, material and coefficients
        cell_id = words[1]
        cell_universe = words[2]
        if words[3] == 'fill':
            coefficients            = [str(x) for x in words[5:]]
            for x in range(len(coefficients)-1, 0, -1):
                if coefficients[x]=='-':
                    coefficients[x+1]=(f'-{coefficients[x+1]}')
                    del(coefficients[x])
            coefficients = ' '.join(coefficients)
            for name, surface_id in sorted(name_to_id.items(), key=lambda x: len(x[0]), reverse=True):
                coefficients = coefficients.replace(name, str(surface_id))
            try:
                cell_region  = openmc.Region.from_expression(expression = coefficients, surfaces = openmc_surfaces)
            except Exception:
                print(f'Failed on line: {line}')
                raise
            if words[4] in openmc_universes:
                openmc_cells[cell_id]        = openmc.Cell(fill=openmc_universes[words[4]], region=cell_region)
                openmc_universes[cell_universe].add_cell(openmc_cells[cell_id])
            elif words[4] in openmc_lattices:
                openmc_cells[cell_id]        = openmc.Cell(fill=openmc_lattices[words[4]], region=cell_region)
                openmc_universes[cell_universe].add_cell(openmc_cells[cell_id])


#---------------------------------------------------------------------------------------------------------------------------

with open('materials_converted', 'w') as fo:
    fo.write(str(openmc_materials) + '\n')

with open('lattice_converted', 'w') as fo:
    fo.write(str(openmc_lattices) + '\n')

with open('universe_converted', 'w') as fo:
    fo.write(str(openmc_universes) + '\n')

with open('cell_converted', 'w') as fo:
    fo.write(str(openmc_cells) + '\n')

with open('surface_converted', 'w') as fo:
    fo.write(str(openmc_surfaces) + '\n')

#------------------------------------Settings-----------------------------------------------
mate = []
materials = list(openmc_materials)
for name in materials:
    mate.append(openmc_materials[name])
material = openmc.Materials(mate)
material.export_to_xml()

geometry             = openmc.Geometry(openmc_universes['0'])
geometry.export_to_xml()


point                = openmc.stats.Point((0, 0, 0))
source               = openmc.Source(space=point)

settings             = openmc.Settings()
settings.source      = source
settings.batches     = 130
settings.inactive    = 30
settings.particles   = 10000
settings.temperature = {'method': 'interpolation'}

settings.export_to_xml()
