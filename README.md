# pyVDERM

Volumetric Density-Equalizing Reference Map - A Python implementation of the VDERM algorithm for 3D shape deformation.

## Overview

pyVDERM implements the Volumetric Density-Equalizing Reference Map (VDERM) method by [Choi & Rycroft (2020)](https://link.springer.com/article/10.1007/s10915-021-01411-4). VDERM is a 3D generalization of the diffusion-based cartogram method, enabling volume-preserving deformations of 3D objects based on prescribed density distributions.

### Applications

- 3D data visualization and cartograms
- Adaptive mesh refinement
- Shape modeling and morphing

### Key Features

- Fast regular grid interpolation
- Comprehensive visualization tools with matplotlib animations
- Flexible export options (XYZ, STL, VTK for ParaView)
- Optional mesh support via PyMeshLab
- Progress tracking and intermediate state exports
- Automatic grid sizing with customizable padding

## Installation

### Basic Installation (Point Cloud Operations Only)
```bash
pip install pyVDERM
```

This installs the core VDERM algorithm with support for `.xyz` point cloud files.

### With Mesh Support

For full functionality including mesh I/O and Poisson reconstruction:
```bash
pip install pyVDERM[mesh]
```

This adds PyMeshLab for reading mesh files (STL, OBJ, PLY) and reconstructing meshes from point clouds.

### Development Installation
```bash
git clone https://github.com/yourusername/pyVDERM.git
cd pyVDERM
pip install -e .[mesh]
```

## Quick Start
```python
import pyVDERM as vd
import numpy as np

# 1. Load a mesh and create surface point cloud
surface_points, normals = vd.create_pcd('mesh.stl', n_pts=25000)

# 2. Create computational grid (automatically sized)
params = vd.make_initial_grid(surface_points, max_points=32768)

# 3. Initialize VDERM grid
vderm_grid = vd.VDERMGrid(
    shape=params['shape'],
    h=params['h'],
    min_bounds=params['min_bounds']
)

# 4. Define density field (controls the deformation)
def my_density(x, y, z):
    r = np.sqrt((x - 1.5)**2 + (y - 1.5)**2 + (z - 1.5)**2)
    return 1.0 + 3.0 * np.exp(-5 * r**2)

vderm_grid.set_density(my_density)

# 5. Run deformation
deformed_grid = vd.run_VDERM(vderm_grid, n_max=100, max_eps=0.02)

# 6. Interpolate deformation to surface
final_surface = vd.interpolate_to_surface(
    surface_points,
    params,
    deformed_grid.get_displacement_field()
)

# 7. Export results
vd.export_mesh('deformed_mesh.stl', final_surface, depth=8)
```

## Examples

Detailed Jupyter notebook examples are available in the `examples/` directory:

- **01_quickStart.ipynb**: Basic workflow and concepts
- **02_boundaryConditions.ipynb**: Understanding and using boundary conditions
- **03_densityFields.ipynb**: Different density functions and their effects
- **04_tracking.ipynb**: Creating animations and tracking a deformation on a non-trivial object
- **05_pyVDERMlite.ipynb**: Point-cloud-only workflow without mesh dependencies

## File Formats

### XYZ Format (Space-delimited text)

pyVDERM uses flexible XYZ files that automatically adapt based on available data:
```
# Positions only (3 columns)
x y z

# Positions + densities (4 columns)
x y z rho

# Positions + normals/velocities (6 columns)
x y z n_x n_y n_z

# Complete (7 columns)
x y z n_x n_y n_z rho
```

Functions `read_xyz()` and `write_xyz()` automatically detect and handle these formats.

## Tips and Best Practices

### Choosing Grid Resolution

- **Small objects or quick tests**: 15,000-30,000 points (20-30³)
- **Standard resolution**: 30,000-50,000 points (30-35³)
- **High quality**: 100,000-250,000 points (45-60³)

Higher resolution gives smoother results but increases computation time.

### Density Field Design

For smooth, predictable deformations:
- Keep densities positive: ρ > 0
- Keep sharp discontinuities 2-3 grid cells away from surface of the object
- When possible, keep large density gradients embedded in a uniform density sea, rather than against a fixed boundary

### Boundary Conditions

The algorithm uses no-flux boundary conditions via ghost nodes:
- Density doesn't leak through boundaries
- Boundaries can still move slightly (typically << 1 grid cell)
- Use padding to minimize boundary effects unless a fixed boundary is needed

### Numerical Stability

If you encounter instability (epsilon becoming very large or negative):

1. Try smaller timestep: `vd.run_VDERM(grid, dt=0.001)`
2. Check your density field for extreme gradients
3. Increase grid resolution

For most cases, automatic timestep selection works well.

## Dependencies

### Required
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.3
- pandas >= 1.3
- tqdm >= 4.60

### Optional (but recommended)
- pymeshlab >= 2023.12 (for mesh I/O and Poisson reconstruction)

## Citation

If you use this package in academic work, please cite the original VDERM paper:
```bibtex
@article{choi2021volumetric,
  title={Volumetric density-equalizing reference map with applications},
  author={Choi, Gary Pui-Tung and Rycroft, Chris H},
  journal={Journal of Scientific Computing},
  volume={86},
  number={3},
  pages={1--26},
  year={2021},
  publisher={Springer}
}
```
And optionally, this implementation:
```bibtex
@software{vderm2026,
  title={pyVDERM: A Python implementation of Volumetric Density-Equalizing Reference Map},
  author={Jonah Spector},
  year={2026},
  url={https://github.com/yourusername/vderm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original VDERM algorithm by Gary P.T. Choi and Chris H. Rycroft
- Based on the diffusion cartogram method by Gastner & Newman (2004)

## Support

- Documentation: [GitHub Wiki](https://github.com/jspector792/pyVDERM/wiki)
- Issues: [GitHub Issues](https://github.com/jspector792/pyVDERM/issues)
