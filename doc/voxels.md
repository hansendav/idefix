Documentation and developer's notes on voxels in IDEFIX package.

Main concepts
=============

3D Images
---------

### Image referential system

Similarly to 2D images of go referenced data, the 3D images coordinates system
does not directly correspond to the one contained in the LiDAR point clouds.

When comparing NumPy and geographic coordinate systems for 2D raters we can
notice these orientations:

```
NumPy       Geophic
  y      
 +-->         ^
x|           y|
 V            +-->
                x
 Coordinate systems
```

For 3D data, the NumPy coordinate respect the right hand rule while the LiDAR
*z* axis will directly correspond to altitude, thus breaking the right hand
rule.

IDEFIX silently project the data from one system to another while providing
utility function to get back any projection or axis labels.

Voxels data
---

- Point cloud arrangement
    + **Density**
    + Orientation PCA
    + Normals
- Feature distribution
    + **Mean**: Reduce noise in the cell
    + Variance: Characterize the distribution of the data
    + **Mode**: Return the majority value (e.g. labels)
    + Entropy
    + *Min*: To create last echoes map
    + Max
    + Quantil

Annexes
=======

Voxels glossary
---

- Step, resolution, bins...

Algorithm
---

### Mode

Insight and warn when use with high entropic data.

- Slow (usual mode)
- Quick (versus boost)

Ideas
---

- Unified voxel format
    + Do not store complete matrix
        * Point cloud like to benefit of PC IO for import/export.
        * Matrices always start to 0,0,0 image like, need metadata (start,
          steps, dtypes, field names...)
    + Allow matrice aumgentation

