# SPDX-FileCopyrightText: 2023-2024 UChicago Argonne, LLC
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import re
import shlex
from typing import List, Set, Union, Dict

import numpy as np
import openmc
from openmc.data import get_thermal_name
from openmc.data.ace import get_metadata

from .serpent_geometry import hexxc, hexyc, sqc, vertical_stack


INPUT_KEYWORDS = [
    'branch',
    'casematrix',
    'cell',
    'coef',
    'datamesh',
    'dep',
    'det',
    'div',
    'dtrans',
    'ene',
    'ftrans',
    'fun',
    'hisv',
    'ifc',
    'include',
    'lat',
    'mat',
    'mesh',
    'mflow',
    'mix',
    'nest',
    'particle',
    'pbed',
    'phb',
    'pin',
    'plot',
    'rep',
    'sample',
    'sens',
    'set',
    'solid',
    'src',
    'strans',
    'surf',
    'therm',
    'thermstoch',
    'tme',
    'trans',
    'transa',
    'transb',
    'transv',
    'umsh',
    'utrans',
    'wwgen',
    'wwin'
]

MATERIAL_KEYWORD_PARAMS = {
    'tmp': 2,
    'tms': 2,
    'tft': 3,
    'rgb': 4,
    'vol': 2,
    'mass': 2,
    'burn': 2,
    'fix': 3,
    'moder': 3
}


#---------------------------------------------------------------------
# Definig a new surface ID with all integers
def update_openmc_surfaces(openmc_surfaces):
    # Determine maximum integer ID
    max_id = 0
    for name in openmc_surfaces.keys():
        if name.isnumeric():
            max_id = max(max_id, int(name))

    # Change non-numeric keys to numeric
    strid_to_intid = {}
    for name in openmc_surfaces.keys():
        if not name.isnumeric():
            max_id += 1
            strid_to_intid[name] = max_id

    for str_id, int_id in strid_to_intid.items():
        openmc_surfaces[str(int_id)] = openmc_surfaces.pop(str_id)

    return strid_to_intid


def expand_include_cards(lines: List[str]) -> List[str]:
    """Replace all 'include' cards"""
    index = 0
    while True:
        # If we've reached end of lines, return
        if index >= len(lines):
            return lines

        # Get words in current line
        words = shlex.split(lines[index])

        if words and first_word(words) == 'include':
            # Read lines from included file
            include_path = Path(words[1])
            with include_path.open('r') as fh:
                insert_lines = fh.readlines()

            # Replace current line with ones from file
            lines[index:index + 1] = insert_lines
        else:
            index += 1


def remove_comments(lines: List[str]) -> List[str]:
    """Remove comments and empty lines"""
    text = ''.join(lines)

    # Remove comments
    text = re.sub('%.*$', '', text, flags=re.MULTILINE)

    # Remove empty lines
    lines = [line for line in text.splitlines(keepends=True) if not line.isspace()]

    # TODO: Remove C-style comments: /*    */
    return lines


def first_word(input: Union[str, List[str]]) -> str:
    """Get lowercased first word from a line or list of words"""
    words = input.split() if isinstance(input, str) else input
    return words[0].lower()


def join_lines(lines: List[str], cards: Set[str]) -> List[str]:
    """Join input for a single card over multiple lines into a single line"""
    index = 0
    while True:
        # If we've reached end of lines, return
        if index >= len(lines):
            return lines

        if first_word(lines[index]) in cards:
            while index + 1 < len(lines):
                if first_word(lines[index + 1]) in INPUT_KEYWORDS:
                    break
                lines[index] += lines.pop(index + 1)

        index += 1


def parse_therm_cards(lines: List[str]) -> Dict[str, str]:
    """Parse 'therm' cards"""
    therm_materials  = {}
    for line in lines:
        words = line.split()
        if first_word(words) != 'therm':
            continue

        if len(words) > 3:
            therm_data = {'name': words[1], 'temp': words[2], 'lib': words[3:]}
        else:
            therm_data = {'name': words[1], 'lib': words[2:]}

        name = therm_data['lib'][0]
        if '.' in name:
            name, _ = name.split('.')
        therm_materials[therm_data['name']] = get_thermal_name(name)

    return therm_materials


def parse_mat_mix_cards(lines: List[str], therm_materials: Dict[str, str]) -> Dict[str, openmc.Material]:
    """Parse 'mat' and 'mix' cards"""

    # TODO: Fix material ID assignment to match Serpent

    openmc_materials = {}
    for line in lines:
        words = line.split()
        keyword = first_word(words)

        if keyword == 'mat':
            name = words[1]
            density = words[2]
            openmc_materials[name] = mat = openmc.Material(name=name)
            if density != 'sum':
                density = float(density)
                if density < 0:
                    mat.set_density('g/cm3', abs(density))
                else:
                    mat.set_density('atom/b-cm', density)

            # Read through keywords on mat card
            index = 3
            while True:
                mat_keyword = words[index].lower()
                if mat_keyword in MATERIAL_KEYWORD_PARAMS:
                    if mat_keyword == 'tmp':
                        mat.temperature = float(words[index + 1])
                    elif mat_keyword == 'moder':
                        mat.add_s_alpha_beta(therm_materials[words[index + 1]])
                    # TODO: Handle other keywords
                    index += MATERIAL_KEYWORD_PARAMS[mat_keyword]
                else:
                    break

            # Read nuclides and fractions
            for nuclide, percent in zip(words[index::2], words[index+1::2]):
                percent = float(percent)
                if '.' in nuclide:
                    zaid, xs = nuclide.split('.')
                    if '-' in zaid:
                        element, A = zaid.split('-')
                        if 'm' in A[-1]:
                            A = A.translate( { ord("m"): None } )
                            zaid = zaid[:-1]
                            print(f'Warning! "m" is ignored for {nuclide}')
                        A = int(A)
                        name = zaid.translate( { ord("-"): None } )
                    else:
                        name, element, Z, A, metastable = get_metadata(int(zaid))
                else:
                    if '-' in zaid:
                        element, A = zaid.split('-')
                        if 'm' in A[-1]:
                            A = A.translate( { ord("m"): None } )
                            zaid = zaid[:-1]
                            print(f'Warning! "m" is ignored for {nuclide}')
                        A = int(A)
                        name = zaid.translate( { ord("-"): None } )
                    else:
                        zaid = nuclide
                        name, element, Z, A, metastable = get_metadata(int(zaid))
                if A > 0:
                    mat.add_nuclide(name, abs(percent), 'wo' if percent < 0 else 'ao')
                else:
                    mat.add_element(element, abs(percent), 'wo' if percent < 0 else 'ao')

        elif keyword == 'mix':
            mat_id = words[1]
            # TODO: Account for rgb, vol, mass keywords
            mix = [openmc_materials[mix_id] for mix_id in words[2::2]]
            mix_per = [float(percent)/100 for percent in words[3::2]]
            openmc_materials[mat_id] = openmc.Material.mix_materials(
                mix, mix_per, 'vo' if mix_per[0] > 0 else 'wo')

    return openmc_materials


def parse_set_cards(lines: List[str]) -> Dict[str, List[str]]:
    """Parse input options on 'set' cards."""
    options = {}
    for line in lines:
        words = line.split()
        if first_word(words) == 'set':
            options[words[1]] = words[2:]
    return options


def parse_surf_cards(lines: List[str]) -> Dict[str, openmc.Surface]:
    """Parse 'surf' cards"""
    openmc_surfaces = {}
    for line in lines:
        words = line.split()
        if first_word(words) == 'surf':
            # Read ID, surface type and coefficients
            _, name, surface_type, *coefficients = words
            coefficients = [float(x) for x in coefficients]

            # Convert to OpenMC surface and add to dictionary
            if surface_type == 'px':
                openmc_surfaces[name] = openmc.XPlane(coefficients[0])
            elif surface_type == 'py':
                openmc_surfaces[name] = openmc.YPlane(coefficients[0])
            elif surface_type == 'pz':
                openmc_surfaces[name] = openmc.ZPlane(coefficients[0])
            elif surface_type in ('cyl', 'cylz'):
                if len(coefficients) == 3:
                    x0, y0, r = coefficients
                    openmc_surfaces[name] = openmc.ZCylinder(x0, y0, r)
                elif len(coefficients) == 5:
                    x0, y0, r, z0, z1 = coefficients
                    center_base = (x0, y0, z0)
                    height = z1 - z0
                    radius = r
                    openmc_surfaces[name] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='z')
            elif surface_type == 'cylx':
                if len(coefficients) == 3:
                    y0, z0, r = coefficients
                    openmc_surfaces[name] = openmc.XCylinder(y0, z0, r)
                elif len(coefficients) == 5:
                    y0, z0, r, x0, x1 = coefficients
                    center_base = (x0, y0, z0)
                    height = x1 - x0
                    radius = r
                    openmc_surfaces[name] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='x')
            elif surface_type == 'cyly':
                if len(coefficients) == 3:
                    x0, z0, r = coefficients
                    openmc_surfaces[name] = openmc.YCylinder(x0, z0, r)
                elif len(coefficients) == 5:
                    x0, z0, r, y0, y1 = coefficients
                    center_base = (x0, y0, z0)
                    height = y1 - y0
                    radius = r
                    openmc_surfaces[name] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='y')
            elif surface_type == 'sqc':
                x0, y0, half_width = coefficients
                openmc_surfaces[name] = sqc(x0, y0, half_width)
            elif surface_type == 'torx':
                x0, y0, z0, A, B, C = coefficients
                openmc_surfaces[name] = openmc.XTorus(x0, y0, z0, A, B, C)
            elif surface_type == 'tory':
                x0, y0, z0, A, B, C = coefficients
                openmc_surfaces[name] = openmc.YTorus(x0, y0, z0, A, B, C)
            elif surface_type == 'torz':
                x0, y0, z0, A, B, C = coefficients
                openmc_surfaces[name] = openmc.ZTorus(x0, y0, z0, A, B, C)
            elif surface_type == 'sph':
                x0, y0, z0, r = coefficients
                openmc_surfaces[name] = openmc.Sphere(x0, y0, z0, r)
            elif surface_type == 'plane':
                A, B, C, D = coefficients
                openmc_surfaces[name] = openmc.Plane(A, B, C, D)
            elif surface_type == 'cone':
                x0, y0, z0, r, h = coefficients
                R = coefficients[3]/coefficients[4]
                Z0 = coefficients[4] + coefficients[2]
                up = (h < 0)
                openmc_surfaces[name] = openmc.model.ZConeOneSided(x0, y0, Z0, R, up)
            elif surface_type == 'hexxc':
                x0, y0, d = coefficients
                openmc_surfaces[name] = hexxc(x0, y0, d)
            elif surface_type == 'hexyc':
                x0, y0, d = coefficients
                openmc_surfaces[name] = hexyc(x0, y0, d)
            else:
                raise ValueError(f"Surface type '{surface_type}' not yet supported.")

    # Conversion of string surface ids to integers ids
    name_to_id = update_openmc_surfaces(openmc_surfaces)
    keys = list(openmc_surfaces.keys())
    for key in keys:
        openmc_surfaces[int(key)] = openmc_surfaces.pop(key)

    return openmc_surfaces, name_to_id


def main():
    openmc_cells     = {}
    openmc_universes = {}
    openmc_lattices  = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    args = parser.parse_args()

    # Read lines from input file
    with args.input_file.open('r') as fh:
        all_lines = fh.readlines()

    # Preprocessing steps: replace 'include' cards, remove comments and empty
    # lines, join cards over multiple lines
    all_lines = expand_include_cards(all_lines)
    all_lines = remove_comments(all_lines)
    all_lines = join_lines(all_lines, {'therm', 'mat', 'mix', 'set', 'surf'})

    # Read thermal scattering cards
    therm_materials = parse_therm_cards(all_lines)

    # Read material and mixture cards
    openmc_materials = parse_mat_mix_cards(all_lines, therm_materials)

    # Read input options on 'set' cards
    options = parse_set_cards(all_lines)

    # Read surfaces on 'surf' cards
    openmc_surfaces, name_to_id = parse_surf_cards(all_lines)

    #--------------------------------------------------------------------------------
    #Conversion of a SERPENT cell and universe to a OpenMC cell and universe
    outer_surfaces = []
    inner_surfaces = []
    for line in all_lines:
        words = line.split()
        if words[0] == 'cell':
            # Creating an outside universe for the lattice outside
            openmc_universes['outside'] = openmc.Universe()

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
                openmc_universes['outside'].add_cell(openmc_cells[cell_id])
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

    boundary = options['bc'][0] if 'bc' in options else None
    for surface in outer_surfaces:
        if '-' in surface:
            surface = surface[1:]
        # TODO: Handle all variations of 'set bc'
        if boundary == '1':
            boundary = 'vacuum'
        elif boundary == '2':
            boundary = 'reflective'
        elif boundary == '3':
            boundary = 'periodic'
        openmc_surfaces[int(surface)].boundary_type = boundary

    #--------------------Pin definition---------------------------------------
    ctrl = 'ctrl'
    surfaces = []
    items = []
    for line in all_lines:
        words = line.split()

        # Read ID, universe, material and coefficients
        if words and words[0] == 'pin':
            cell_universe = words[1]
            ctrl = words[0]
            surfaces = []
            items = []
        else:
            if words and (words[0] in ('surf', 'cell', 'mat', 'lat', 'set', 'include', 'plot', 'therm', 'dep')):
                ctrl = words[0]
            elif len(words) > 1 and ctrl == 'pin' and words[0] != 'pin':
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
                if surfaces == []:
                    cell = openmc.Cell(fill=items)
                    openmc_universes[cell_universe] = openmc.Universe(cells=[cell])
                else:
                    openmc_universes[cell_universe] = openmc.model.pin(surfaces, items, subdivisions=None, divide_vols=True)

    # NOTE: If there is only one material and no surface, this code does not work. Needs to be fixed
    #---------------------------------------------------------------------------------------------------------------------------
    #Conversion of a SERPENT lattice to a OpenMC lattice
    lattice_type = ''
    number_of_rings = 0
    for line in all_lines:
        words = line.split()

        if words[0] == 'lat':
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
                x0, y0                                  = words[3:5]
                nx, ny                                  = words[5:7]
                pitch                                   = float(words[7])
                number_of_rings                         = int((float(nx)+1)/2)
                #print(number_of_rings)
                openmc_lattices[lattice_id]             = openmc.HexLattice()
                #print(openmc.HexLattice.show_indices(int(number_of_rings)))
                openmc_lattices[lattice_id].orientation = 'x'
                openmc_lattices[lattice_id].center  = float(x0), float(y0)
                openmc_lattices[lattice_id].pitch       = [pitch]
            elif lattice_type == '3':
                x0, y0                                  = words[3:5]
                nx, ny                                  = words[5:7]
                pitch                                   = float(words[7])
                number_of_rings                         = (float(nx)+1)/2
                openmc_lattices[lattice_id]             = openmc.HexLattice(number_of_rings)
                #print(openmc_lattices[lattice_id].show_indices())
                openmc_lattices[lattice_id].orientation = 'y'
                openmc_lattices[lattice_id].lower_left  = (-(float(nx)/2)*pitch, -(float(ny)/2)*pitch)
                openmc_lattices[lattice_id].pitch       = [pitch]
            elif lattice_type == '6':
                openmc_lattices[lattice_id]             = openmc.RectLattice()
                x0, y0                                  = words[3:5]
                pitch                                   = float(words[5])
                openmc_lattices[lattice_id].lower_left  = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch       = (pitch, pitch)
            elif lattice_type == '7':
                openmc_lattices[lattice_id]             = openmc.HexLattice()
                openmc_lattices[lattice_id].orientation = 'x'
                x0, y0                                  = words[3:5]
                pitch                                   = float(words[5])
                openmc_lattices[lattice_id].lower_left  = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch       = [pitch]
            elif lattice_type == '8':
                openmc_lattices[lattice_id]             = openmc.HexLattice()
                openmc_lattices[lattice_id].orientation = 'y'
                x0, y0                                  = words[3:5]
                pitch                                   = float(words[5])
                openmc_lattices[lattice_id].lower_left  = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch       = [pitch]
            elif lattice_type == '9':
                x0, y0                                  = words[3:5]
                x0, y0                                  = float(x0), float(y0)
                n                                       = float(words[5])
            # Missing circular cluster array (4), 3D cuboidal lattice (11), x-type triangular lattice (14)
            # 3D x-type hexagonal prism lattice (12), and 3D y-type hexagonal prism lattice (13)
            elif lattice_type == '4':
                raise ValueError('Lattice geometry: circular cluster array is not supported!')
            elif lattice_type == '11':
                raise ValueError('Lattice geometry: 3D cuboidal lattice is not supported!')
            elif lattice_type == '12':
                raise ValueError('Lattice geometry: 3D x-type hexagonal prism lattice is not supported!')
            elif lattice_type == '13':
                raise ValueError('Lattice geometry: 3D y-type hexagonal prism lattice is not supported!')
            elif lattice_type == '14':
                raise ValueError('Lattice geometry: x-type triangular lattice is not defined supported!')

            # !!!!!!Think about it again!!!!!!!!! Does it work if we have multiple lattice geometries
            # openmc_lattices[lattice_id].outer = openmc_universes['outside']

        if words[0] in ('surf', 'cell', 'mat', 'lat', 'set', 'include', 'plot', 'therm', 'dep'):
            ctrl = words[0]
            #print(ctrl)

        if words[0] not in ('surf', 'cell', 'lat', 'plot', 'set') and lattice_type == '9':
            z0.append(float(words[0]))
            uni.append(openmc_universes[words[1]])
            openmc_lattices[lattice_id]                 = vertical_stack(z0, uni, x0, y0)
        elif words[0] not in ('surf', 'mat', 'cell', 'lat', 'set', 'include', 'plot', 'therm', 'pin', 'dep') and ctrl != 'mat' and ctrl != 'dep' and lattice_type !='9' and words[0] not in openmc_materials:
            if lattice_type == '6' or lattice_type == '1':
                for x in range(len(words)):
                    words[x]                                = openmc_universes[words[x]]
                control                                     = words
                uni.append(control)
                lattice                                     = openmc_lattices[lattice_id]
                lattice.universes                           = list(reversed(uni))
            elif lattice_type == '2':
                uni.append(words)
                if len(uni) == int(nx):
                    for n in range(int(nx)):
                        if n < number_of_rings:
                            uni[n] = uni[n][-(number_of_rings+n):]
                        elif n >= number_of_rings:
                            uni[n] = uni[n][:(number_of_rings-(n+1))]

    # !!!!!This is NOT going to work with the multiple lattices in one geometry
    # Conversion of hexagonal lattice geometry
    if number_of_rings != 0:
        rings = []
        ctrl = number_of_rings
        nx = int(nx)
        for r in range(number_of_rings):
            ring = []
            x = 0
            y = 0
            for item in range(int((nx-1)*3)):
                # starting point
                if item == 0:
                    a, b = number_of_rings-1, nx-1
                    ring.append(uni[a][b])
                # till first corner
                elif item <= ctrl-1:
                    x, y = x+1, y-1
                    a, b = number_of_rings-1+x, nx-1+y
                    ring.append(uni[a][b])
                # till second corner
                elif item > ctrl-1 and item <= 2*(ctrl-1):
                    y = y-1
                    a, b = number_of_rings-1+x, nx-1+y
                    ring.append(uni[a][b])
                # till third and fourth corners
                elif item > 2*(ctrl-1) and item <= 4*(ctrl-1):
                    x = x-1
                    a, b = number_of_rings-1+x, nx-1+y
                    ring.append(uni[a][b])
                # till fifth corner
                elif item > 4*(ctrl-1) and item <= 5*(ctrl-1):
                    y = y+1
                    a, b = number_of_rings-1+x, nx-1+y
                    ring.append(uni[a][b])
                # after fifth corner
                elif item > 5*(ctrl-1)and item < 6*(ctrl-1):
                    x, y = x+1, y+1
                    a, b = number_of_rings-1+x, nx-1+y
                    ring.append(uni[a][b])
            rings.append(ring)
            for i in range(len(ring)):
                ring[i] = openmc_universes[ring[i]]
            name = f'{r+1}.ring'
            ctrl = ctrl - 1
            nx = nx - 1
        #print(rings)
        for r in range(len(rings)):
            openmc_lattices[lattice_id].universes = rings

    #---------------------------------------------------------------------------------------------------------------------------
    # Creating cells with 'fill' command
    for line in all_lines:
        words = line.split()
        if words[0] == 'cell':
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


    #------------------------------------Settings-----------------------------------------------

    model = openmc.Model()
    model.geometry = openmc.Geometry(openmc_universes['0'])
    model.materials = openmc.Materials(openmc_materials.values())

    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    model.settings.batches = 130
    model.settings.inactive = 30
    model.settings.particles = 10000
    model.settings.temperature = {'method': 'interpolation'}

    model.export_to_model_xml('model.xml')
