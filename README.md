# VDERM

Volumetric Density-Equalizing Reference Map - A Python implementation

## Installation

### Development mode
```bash
git clone https://github.com/yourusername/vderm.git
cd vderm
pip install -e .
```

### With mesh support
```bash
pip install -e .[mesh]
```

## Quick Start
```python
import vderm

# Load mesh and create surface point cloud
surface_pts, normals = vderm.create_pcd('mesh.stl', n_pts=25000)

# Create computational grid
grid, params = vderm.make_initial_grid(surface_pts, max_points=32768)

# Set up VDERM
vderm_grid = vderm.VDERMGrid(params['shape'], params['h'], params['min_bounds'])
# ... set density field ...

# Run deformation
final_grid = vderm.run_VDERM(vderm_grid, n_max=100)

# Export result
final_surface = final_grid.interpolate_to_points(surface_pts)
vderm.export_mesh_file('output.stl', final_surface)
```

## Citation

Based on the VDERM method by Choi & Rycroft (2020).