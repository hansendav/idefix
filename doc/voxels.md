Documentation and developer's notes on voxels in IDEFIX package.

Voxels data
---

- Density
- Distribution
    + Mean
    + Variance
    + Mode
    + Entropy
    + Min
    + Max
    + Quantil

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

