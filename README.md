# Serpent Conversion Tools for OpenMC

[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

This repository provides tools for parsing/converting Serpent models to OpenMC
classes and/or XML files. To install these tools, run:

    python -m pip install git+https://github.com/openmc-dev/openmc_serpent_adapter.git

This makes the `openmc_serpent_adapter` Python module and `serpent_to_openmc`
console script available. To convert an Serpent model, run:

    serpent_to_openmc serpent_input

## Disclaimer

There has been no methodical V&V on this converter; use at your own risk!

## Known Limitations

The converter currently only handles geometry and material information; source
definition (`src`) and detectors (`det`) are ignored.

Many geometry features are not currrently supported:

- Many lattice types
- Transformations (`trans`, `ftrans`, `strans`, `transa`, `transv`)
- Nested universe definitions (`nest`)
- Particle geometry (`particle`)
- Explicit stochastic geometry (`pbed`)
- Irregular 3D geometry (`solid`)
- Unstructured mesh-based geometry (`umsh`)
- Stochastic Voronoi tessellation geomery (`voro`)
