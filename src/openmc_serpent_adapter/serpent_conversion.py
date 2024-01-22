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


def expand_include_cards(lines: List[str]) -> List[str]:
    """Replace all 'include' cards"""
    index = 0
    while True:
        # If we've reached end of lines, return
        if index >= len(lines):
            return lines

        # Get words in current line
        words = lines[index].split()

        if words and first_word(words) == 'include':
            # Read lines from included file. Need to use shlex splitting to
            # handle paths with spaces embedded
            include_path = Path(shlex.split(lines[index])[1])
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


def _get_max_numeric_id(lines: List[str], keywords: Set[str], position: int = 1) -> int:
    max_id = -1
    for line in lines:
        words = line.split()
        if first_word(words) in keywords:
            name = words[position]
            if name.isnumeric():
                max_id = max(max_id, int(name))
    return max_id


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

    # Avoid clasing with numeric IDs from Serpent
    openmc.Material.next_id = _get_max_numeric_id(lines, {'mat', 'mix'}) + 1

    openmc_materials = {}
    for line in lines:
        words = line.split()
        keyword = first_word(words)

        if keyword == 'mat':
            name = words[1]
            material_id = int(name) if name.isnumeric() else None
            density = words[2]
            openmc_materials[name] = mat = openmc.Material(
                name=name, material_id=material_id)
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
            name = words[1]
            # TODO: Account for rgb, vol, mass keywords
            mix = [openmc_materials[mix_id] for mix_id in words[2::2]]
            mix_per = [float(percent)/100 for percent in words[3::2]]
            openmc_materials[name] = openmc.Material.mix_materials(
                mix, mix_per, 'vo' if mix_per[0] > 0 else 'wo')
            if name.isnumeric():
                openmc_materials[name].id = int(name)

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

    # Avoid clasing with numeric IDs from Serpent
    openmc.Surface.next_id = _get_max_numeric_id(lines, {'surf'}) + 1

    openmc_surfaces = {}
    for line in lines:
        words = line.split()
        if first_word(words) == 'surf':
            # Read ID, surface type and coefficients
            _, name, surface_type, *coefficients = words
            uid = int(name) if name.isnumeric() else None
            coefficients = [float(x) for x in coefficients]
            kwargs = {'name': name, 'surface_id': uid}

            # Convert to OpenMC surface and add to dictionary
            if surface_type == 'px':
                openmc_surfaces[name] = openmc.XPlane(coefficients[0], **kwargs)
            elif surface_type == 'py':
                openmc_surfaces[name] = openmc.YPlane(coefficients[0], **kwargs)
            elif surface_type == 'pz':
                openmc_surfaces[name] = openmc.ZPlane(coefficients[0], **kwargs)
            elif surface_type in ('cyl', 'cylz'):
                if len(coefficients) == 3:
                    x0, y0, r = coefficients
                    openmc_surfaces[name] = openmc.ZCylinder(x0, y0, r, **kwargs)
                elif len(coefficients) == 5:
                    x0, y0, r, z0, z1 = coefficients
                    center_base = (x0, y0, z0)
                    height = z1 - z0
                    radius = r
                    openmc_surfaces[name] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='z')
            elif surface_type == 'cylx':
                if len(coefficients) == 3:
                    y0, z0, r = coefficients
                    openmc_surfaces[name] = openmc.XCylinder(y0, z0, r, **kwargs)
                elif len(coefficients) == 5:
                    y0, z0, r, x0, x1 = coefficients
                    center_base = (x0, y0, z0)
                    height = x1 - x0
                    radius = r
                    openmc_surfaces[name] = openmc.model.RightCircularCylinder(center_base, height, radius, axis='x')
            elif surface_type == 'cyly':
                if len(coefficients) == 3:
                    x0, z0, r = coefficients
                    openmc_surfaces[name] = openmc.YCylinder(x0, z0, r, **kwargs)
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
                openmc_surfaces[name] = openmc.XTorus(x0, y0, z0, A, B, C, **kwargs)
            elif surface_type == 'tory':
                x0, y0, z0, A, B, C = coefficients
                openmc_surfaces[name] = openmc.YTorus(x0, y0, z0, A, B, C, **kwargs)
            elif surface_type == 'torz':
                x0, y0, z0, A, B, C = coefficients
                openmc_surfaces[name] = openmc.ZTorus(x0, y0, z0, A, B, C, **kwargs)
            elif surface_type == 'sph':
                x0, y0, z0, r = coefficients
                openmc_surfaces[name] = openmc.Sphere(x0, y0, z0, r, **kwargs)
            elif surface_type == 'plane':
                A, B, C, D = coefficients
                openmc_surfaces[name] = openmc.Plane(A, B, C, D, **kwargs)
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

    return openmc_surfaces


def parse_pin_cards(lines: List[str], materials: Dict[str, openmc.Material], universes: Dict[str, openmc.Universe]):
    """Parse 'pin' cards"""
    for line in lines:
        words = line.split()
        if first_word(words) != 'pin':
            continue

        universe_name = words[1]

        # Iteratively read materials/universes and outer radii
        fills = []
        surfaces = []
        index = 2
        while True:
            if words[index] == 'fill':
                fills.append(universes[words[index + 1]])
                index += 2
            else:
                fills.append(materials[words[index]])
                index += 1
            if index >= len(words):
                break
            surfaces.append(openmc.ZCylinder(r=float(words[index])))
            index += 1

        # Construct new universe for pin
        if surfaces:
            universes[universe_name] = openmc.model.pin(surfaces, fills)
        else:
            # Special case where no surfaces are given
            cell = openmc.Cell(fill=fills[0])
            universes[universe_name] = openmc.Universe(cells=[cell])


def parse_cell_cards(
        lines: List[str],
        surfaces: Dict[str, openmc.Surface],
        materials: Dict[str, openmc.Material],
        universes: Dict[str, openmc.Universe],
        lattices: Dict[str, openmc.UniverseBase]):
    """Parse 'cell' cards and return a list of cells marked as 'outside'."""

    # Determine mapping of any surface that has a non-numeric name to a unique
    # integer. This is needed when generating a cell region for OpenMC since it
    # can't handle non-numeric surface names in the expression
    starting_id = openmc.Surface.next_id
    name_to_index: Dict[str, int] = {}
    index_to_surface: Dict[int, openmc.Surface] = {}
    for name, surf in surfaces.items():
        if not name.isnumeric():
            name_to_index[name] = index = starting_id
            starting_id += 1
        else:
            index = int(name)
        index_to_surface[index] = surf

    outside_cells = set()
    for line in lines:
        words = line.split()
        if first_word(words) != 'cell':
            continue

        # Read ID, universe, material and coefficients
        name = words[1]
        cell = openmc.Cell(name=name)

        # Add cell to specified universe
        universe_name = words[2]
        if universe_name not in universes:
            universes[universe_name] = openmc.Universe()
        universes[universe_name].add_cell(cell)

        if words[3] == 'fill':
            # Assign universe/lattice fill to cell
            univ_name = words[4]
            if univ_name in universes:
                cell.fill = universes[univ_name]
            elif words[4] in lattices:
                cell.fill = lattices[univ_name]
            else:
                raise ValueError(f"Cell '{name}' is filled with non-existent universe '{univ_name}'")

            coefficients = words[5:]
        elif words[3] == 'void':
            coefficients = words[4:]
        elif words[3] == 'outside':
            coefficients = words[4:]
            outside_cells.add(cell)
        else:
            cell.fill = materials[words[3]]
            coefficients = words[4:]

        # TODO: Should we keep this fixup?
        for x in range(len(coefficients)-1, 0, -1):
            if coefficients[x] == '-':
                coefficients[x+1] = f'-{coefficients[x+1]}'
                del coefficients[x]

        # Creating regions
        coefficient = ' '.join(coefficients)
        for name, index in sorted(name_to_index.items(), key=lambda x: len(x[0]), reverse=True):
            coefficient = coefficient.replace(name, str(index))
        try:
            cell.region = openmc.Region.from_expression(expression=coefficient, surfaces=index_to_surface)
        except Exception:
            raise ValueError(f'Failed to convert cell definition: {line}\n{coefficient}')

    return outside_cells


def determine_boundary_surfaces(geometry: openmc.Geometry, outside_cells: Set[openmc.Cell]) -> List[openmc.Surface]:
    """Determine which surfaces can be used as boundaries based on a set of 'outside' cells."""

    # Determine which halfspaces are associate with outer and inner cells
    outer_halfspaces = set()
    inner_halfspaces = set()
    for cell in geometry.root_universe._cells.values():
        halfspace_ids = {int(s) for s in re.findall(r'-?\d+', str(cell.region))}
        if cell in outside_cells:
            outer_halfspaces |= halfspace_ids
        else:
            inner_halfspaces |= halfspace_ids

    # Eliminate any halfspaces that are used to define inner cells
    outer_halfspaces -= inner_halfspaces

    # The halfspaces that remain involve surfaces that can used as boundaries
    surfaces = geometry.get_all_surfaces()
    return [surfaces[abs(uid)] for uid in outer_halfspaces]


def main():
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
    all_lines = join_lines(all_lines, {'therm', 'mat', 'mix', 'set', 'surf', 'pin', 'cell'})

    # Read thermal scattering cards
    therm_materials = parse_therm_cards(all_lines)

    # Read material and mixture cards
    openmc_materials = parse_mat_mix_cards(all_lines, therm_materials)

    # Read input options on 'set' cards
    options = parse_set_cards(all_lines)

    # Read surfaces on 'surf' cards
    openmc_surfaces = parse_surf_cards(all_lines)

    def get_universe(name):
        if name not in openmc_universes:
            openmc_universes[name] = openmc.Universe()
        return openmc_universes[name]

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
                openmc_lattices[lattice_id] = openmc.RectLattice()
                x0, y0 = words[3:5]
                nx, ny = words[5:7]
                pitch = float(words[7])
                openmc_lattices[lattice_id].lower_left = (-(float(nx)/2)*pitch, -(float(ny)/2)*pitch)
                openmc_lattices[lattice_id].pitch = (pitch, pitch)
            elif lattice_type == '2':
                x0, y0 = words[3:5]
                nx, ny = words[5:7]
                pitch = float(words[7])
                number_of_rings = int((float(nx)+1)/2)
                openmc_lattices[lattice_id] = openmc.HexLattice()
                openmc_lattices[lattice_id].orientation = 'x'
                openmc_lattices[lattice_id].center = float(x0), float(y0)
                openmc_lattices[lattice_id].pitch = [pitch]
            elif lattice_type == '3':
                x0, y0 = words[3:5]
                nx, ny = words[5:7]
                pitch = float(words[7])
                number_of_rings = (float(nx)+1)/2
                openmc_lattices[lattice_id] = openmc.HexLattice(number_of_rings)
                openmc_lattices[lattice_id].orientation = 'y'
                openmc_lattices[lattice_id].lower_left = (-(float(nx)/2)*pitch, -(float(ny)/2)*pitch)
                openmc_lattices[lattice_id].pitch = [pitch]
            elif lattice_type == '6':
                openmc_lattices[lattice_id] = openmc.RectLattice()
                x0, y0 = words[3:5]
                pitch = float(words[5])
                openmc_lattices[lattice_id].lower_left = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch = (pitch, pitch)
            elif lattice_type == '7':
                openmc_lattices[lattice_id] = openmc.HexLattice()
                openmc_lattices[lattice_id].orientation = 'x'
                x0, y0                                  = words[3:5]
                pitch                                   = float(words[5])
                openmc_lattices[lattice_id].lower_left  = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch       = [pitch]
            elif lattice_type == '8':
                openmc_lattices[lattice_id] = openmc.HexLattice()
                openmc_lattices[lattice_id].orientation = 'y'
                x0, y0 = words[3:5]
                pitch = float(words[5])
                openmc_lattices[lattice_id].lower_left = (-(float(x0)+(pitch/2)), -(float(y0)+(pitch/2)))
                openmc_lattices[lattice_id].pitch = [pitch]
            elif lattice_type == '9':
                x0, y0 = words[3:5]
                x0, y0 = float(x0), float(y0)
                n = float(words[5])
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

        if words[0] in ('surf', 'cell', 'mat', 'lat', 'set', 'include', 'plot', 'therm', 'dep'):
            ctrl = words[0]

        if words[0] not in ('surf', 'cell', 'lat', 'plot', 'set') and lattice_type == '9':
            z0.append(float(words[0]))
            uni.append(get_universe(words[1]))
            openmc_lattices[lattice_id] = vertical_stack(z0, uni, x0, y0)
        elif words[0] not in ('surf', 'mat', 'cell', 'lat', 'set', 'include', 'plot', 'therm', 'pin', 'dep') and ctrl != 'mat' and ctrl != 'dep' and lattice_type !='9' and words[0] not in openmc_materials:
            if lattice_type == '6' or lattice_type == '1':
                universes = [get_universe(name) for name in words]
                uni.append(universes)
                lattice = openmc_lattices[lattice_id]
                lattice.universes = list(reversed(uni))
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
            rings.append([get_universe(name) for name in ring])
            name = f'{r+1}.ring'
            ctrl = ctrl - 1
            nx = nx - 1
        #print(rings)
        for r in range(len(rings)):
            openmc_lattices[lattice_id].universes = rings

    # Read pin geometry definitions
    parse_pin_cards(all_lines, openmc_materials, openmc_universes)

    # Read cells on 'cell' cards
    outside_cells = parse_cell_cards(all_lines, openmc_surfaces, openmc_materials,
                                     openmc_universes, openmc_lattices)

    # TODO: Check for 'set root'
    geometry = openmc.Geometry(openmc_universes['0'])

    # Determine what boundary condition to apply based on the 'set bc' card
    boundary = options['bc'][0] if 'bc' in options else None
    # TODO: Handle all variations of 'set bc'
    if boundary == '1':
        boundary = 'vacuum'
    elif boundary == '2':
        boundary = 'reflective'
    elif boundary == '3':
        boundary = 'periodic'

    # Try to infer boundary conditions on surfaces based on which cells were
    # marked as 'outside' cells
    for surf in determine_boundary_surfaces(geometry, outside_cells):
        surf.boundary_type = boundary

    #------------------------------------Settings-----------------------------------------------

    model = openmc.Model(geometry=geometry)
    model.materials = openmc.Materials(openmc_materials.values())

    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    model.settings.batches = 130
    model.settings.inactive = 30
    model.settings.particles = 10000
    model.settings.temperature = {'method': 'interpolation'}

    model.export_to_model_xml('model.xml')
