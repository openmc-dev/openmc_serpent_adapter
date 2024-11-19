# SPDX-FileCopyrightText: 2023-2024 UChicago Argonne, LLC
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import re
import shlex
from typing import List, Tuple, Set, Union, Dict, Any

import numpy as np
import openmc
from openmc.data import get_thermal_name
from openmc.data.ace import get_metadata

from .serpent_geometry import hexxc, hexyc, sqc, vertical_stack, zvessel


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


universe_fill = Union[openmc.UniverseBase, openmc.Lattice]


def expand_include_cards(lines: List[str], basedir: Path) -> List[str]:
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
            if not include_path.is_absolute():
                include_path = basedir / include_path
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


def join_lines(lines: List[str]) -> List[str]:
    """Join input for a single card over multiple lines into a single line"""
    index = 0
    while True:
        # If we've reached end of lines, return
        if index >= len(lines):
            return lines

        if first_word(lines[index]) in INPUT_KEYWORDS:
            while index + 1 < len(lines):
                if first_word(lines[index + 1]) in INPUT_KEYWORDS:
                    break
                lines[index] += lines.pop(index + 1)

        index += 1


def check_unsupported_cards(lines: List[str], keywords: Set[str]):
    """Check for geometry features that are not yet supported."""
    for line in lines:
        keyword = first_word(line)
        if keyword in keywords:
            raise ValueError(f"'{keyword}' card not yet supported.")


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
    therm_materials = {}
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


def parse_trans_cards(lines: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Parse transformations on 'trans' cards."""
    transformations = {}
    for line in lines:
        words = line.split()
        keyword = first_word(words)
        if keyword not in ('trans', 'ftrans', 'strans', 'utrans', 'dtrans'):
            continue

        # Read type, unit, and values
        if keyword == 'trans':
            trans_type = words[1].lower()
            unit = words[2]
            parameters = words[3:]
        elif keyword in ('ftrans', 'strans', 'utrans', 'dtrans'):
            trans_type = keyword[0]
            unit = words[1]
            parameters = words[2:]

        # Check for unsupported transformation type
        if trans_type in ('u', 'l', 'd', 'sr'):
            raise ValueError(f"'trans {trans_type} not yet supported.")

        # Turn values into a dictionary of data that can be more easily used
        if len(parameters) == 1:
            data = {'lvl': parameters[0]}
        elif parameters[0].lower() == 'rot':
            raise ValueError("Transformation with rotation not yet supported.")
        elif len(parameters) == 3:
            data = {'xyz': np.array([float(x) for x in parameters[:3]])}
        else:
            raise ValueError("Transformation with rotation not yet supported.")
        transformations[trans_type, unit] = data

    return transformations


def parse_surf_cards(lines: List[str], transformations: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, openmc.Surface]:
    """Parse 'surf' cards"""

    openmc_surfaces = {}
    for line in lines:
        words = line.split()
        if first_word(words) != 'surf':
            continue

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
            center_base = (x0, y0, z0)
            openmc_surfaces[name] = openmc.model.ConicalFrustum(center_base, (0., 0., h), r, 0.0)
        elif surface_type == 'hexxc':
            x0, y0, d = coefficients
            openmc_surfaces[name] = hexxc(x0, y0, d)
        elif surface_type == 'hexyc':
            x0, y0, d = coefficients
            openmc_surfaces[name] = hexyc(x0, y0, d)
        elif surface_type == 'cuboid':
            xmin, xmax, ymin, ymax, zmin, zmax = coefficients
            openmc_surfaces[name] = openmc.model.RectangularParallelepiped(xmin, xmax, ymin, ymax, zmin, zmax)
        elif surface_type == "vessel":
            if len(coefficients) == 7:
                x0, y0, r, zmin, zmax, hbottom, htop = coefficients
            else:
                x0, y0, r, zmin, zmax, hbottom = coefficients
                htop = hbottom
            openmc_surfaces[name] = openmc.model.Vessel(r, zmin, zmax, hbottom, htop, (x0, y0))
        else:
            raise ValueError(f"Surface type '{surface_type}' not yet supported.")

        # Check for surface transformation
        trans = transformations.get(('s', name))
        if trans is not None and 'xyz' in trans:
            openmc_surfaces[name].translate(trans['xyz'], inplace=True)

    return openmc_surfaces


def parse_lat_cards(lines: List[str], openmc_universes: Dict[str, universe_fill]):
    """Parse 'lat' cards."""

    def get_universe(name: str):
        if name not in openmc_universes:
            uid = int(name) if name.isnumeric() else None
            openmc_universes[name] = openmc.Universe(universe_id=uid)
        return openmc_universes[name]

    for line in lines:
        words = line.split()

        if words[0] != 'lat':
            continue

        universe_name = words[1]
        lattice_id = int(universe_name) if universe_name.isnumeric() else None
        lattice_type = int(words[2])
        if lattice_type in (1, 2, 3, 14):
            # Case I
            x0 = float(words[3])
            y0 = float(words[4])
            nx = int(words[5])
            ny = int(words[6])
            pitch = float(words[7])

            # Put universes into an array
            uni = words[8:]
            universes = np.array([get_universe(name) for name in uni])
            universes.shape = (ny, nx)

            if lattice_type == 1:
                lattice = openmc.RectLattice(lattice_id=lattice_id, name=universe_name)
                lattice.lower_left = (-(nx/2)*pitch, -(ny/2)*pitch)
                lattice.pitch = (pitch, pitch)
                # Set universes and reverse the y direction
                lattice.universes = universes[::-1]
            elif lattice_type in (2, 3):
                lattice = openmc.HexLattice(lattice_id=lattice_id, name=universe_name)
                lattice.orientation = 'x' if lattice_type == 2 else 'y'
                lattice.center = (x0, y0)
                lattice.pitch = [pitch]
                if lattice_type == 2:
                    raise ValueError('Lattice geometry: x-type hexagonal lattice not yet supported.')
                else:
                    raise ValueError('Lattice geometry: y-type hexagonal lattice not yet supported.')
            elif lattice_type == 14:
                raise ValueError('Lattice geometry: x-type triangular lattice not yet supported.')

        elif lattice_type in (6, 7, 8):
            # Case II
            x0 = float(words[3])
            y0 = float(words[4])
            pitch = float(words[5])
            universe = get_universe(words[6])

            if lattice_type == 6:
                lattice = openmc.RectLattice(lattice_id=lattice_id, name=universe_name)
                lattice.lower_left = (-(x0 + pitch/2), -(y0 + pitch/2))
                lattice.pitch = (pitch, pitch)
                lattice.universes = [[universe]]
                lattice.outer = universe
            elif lattice_type in (7, 8):
                lattice = openmc.HexLattice(lattice_id=lattice_id, name=universe_name)
                lattice.orientation = 'x' if lattice_type == 7 else 'y'
                lattice.center = (x0, y0)
                lattice.pitch = [pitch]
                lattice.universes = [[universe]]
                lattice.outer = universes

        elif lattice_type == 4:
            # Case III
            raise ValueError('Lattice geometry: circular cluster array not yet supported!')

        elif lattice_type == 9:
            # Case IV
            x0 = float(words[3])
            y0 = float(words[4])
            n = int(words[5])
            z = [float(x) for x in words[6::2]]
            uni = [get_universe(x) for x in words[7::2]]
            lattice = vertical_stack(z, uni, x0, y0)

        elif lattice_type in (11, 12, 13):
            # Case V
            if lattice_type == 11:
                raise ValueError('Lattice geometry: 3D cuboidal lattice not yet supported!')
            elif lattice_type == 12:
                raise ValueError('Lattice geometry: 3D x-type hexagonal prism lattice not yet supported!')
            elif lattice_type == 13:
                raise ValueError('Lattice geometry: 3D y-type hexagonal prism lattice not yet supported!')

        # Add lattice to dictionary
        openmc_universes[universe_name] = lattice


def parse_pin_cards(lines: List[str], materials: Dict[str, openmc.Material], universes: Dict[str, openmc.Universe]):
    """Parse 'pin' cards"""

    # NOTE: If there is only one material and no surface, this code does not work. Needs to be fixed

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
        transformations: Dict[Tuple[str, str], Dict[str, Any]]):
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
        cell_id = int(name) if name.isnumeric() else None
        cell = openmc.Cell(name=name, cell_id=cell_id)

        # Add cell to specified universe
        universe_name = words[2]
        if universe_name not in universes:
            uid = int(universe_name) if universe_name.isnumeric() else None
            universes[universe_name] = openmc.Universe(universe_id=uid)
        universes[universe_name].add_cell(cell)

        if words[3] == 'fill':
            # Assign universe/lattice fill to cell
            univ_name = words[4]
            if univ_name in universes:
                cell.fill = universes[univ_name]
                trans = transformations.get(('f', name))
                if trans is not None and 'xyz' in trans:
                    cell.translation = trans['xyz']
            else:
                raise ValueError(f"Cell '{name}' is filled with non-existent universe/lattice '{univ_name}'")

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


def serpent_to_model(input_file) -> openmc.Model:
    # Read lines from input file
    input_file = Path(input_file)
    with input_file.open('r') as fh:
        all_lines = fh.readlines()

    # Preprocessing steps: replace 'include' cards, remove comments and empty
    # lines, join cards over multiple lines
    all_lines = expand_include_cards(all_lines, input_file.parent)
    all_lines = remove_comments(all_lines)
    all_lines = join_lines(all_lines)
    check_unsupported_cards(all_lines, {
        'nest', 'particle', 'pbed', 'solid', 'transa', 'transv', 'umsh', 'voro'
    })

    # Avoid clashing with numeric IDs from Serpent. For universes, we need to
    # account for all cards where universe names appear.
    openmc.Material.next_id = _get_max_numeric_id(all_lines, {'mat', 'mix'}) + 1
    openmc.Surface.next_id = _get_max_numeric_id(all_lines, {'surf'}) + 1
    openmc.Cell.next_id = _get_max_numeric_id(all_lines, {'cell'}) + 1
    max_univ1 = _get_max_numeric_id(all_lines, {'lat', 'pin', 'nest', 'particle', 'pbed', 'umsh'})
    max_univ2 = _get_max_numeric_id(all_lines, {'cell', 'solid'}, 2)
    openmc.Universe.next_id = max(max_univ1, max_univ2) + 1
    openmc.Lattice.next_id = max(max_univ1, max_univ2) + 1

    # Read thermal scattering cards
    therm_materials = parse_therm_cards(all_lines)

    # Read material and mixture cards
    openmc_materials = parse_mat_mix_cards(all_lines, therm_materials)

    # Read input options on 'set' cards
    options = parse_set_cards(all_lines)

    # Read transformations
    transformations = parse_trans_cards(all_lines)

    # Read surfaces on 'surf' cards
    openmc_surfaces = parse_surf_cards(all_lines, transformations)

    # Read lattices on 'lat' cards
    openmc_universes = {}
    parse_lat_cards(all_lines, openmc_universes)

    # Read pin geometry definitions
    parse_pin_cards(all_lines, openmc_materials, openmc_universes)

    # Read cells on 'cell' cards
    outside_cells = parse_cell_cards(all_lines, openmc_surfaces, openmc_materials,
                                     openmc_universes, transformations)

    # Create geometry with specified root universe
    root = options['root'][0] if 'root' in options else '0'
    geometry = openmc.Geometry(openmc_universes[root])

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

    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point((0., 0., 0.)))
    model.settings.batches = 130
    model.settings.inactive = 30
    model.settings.particles = 10000
    model.settings.temperature = {'method': 'interpolation'}

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    args = parser.parse_args()

    model = serpent_to_model(args.input_file)
    model.export_to_model_xml('model.xml')
