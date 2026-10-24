# Stanford Bunny curved-mesh benchmark

This experiment uses the Stanford Bunny as a recognizable curved-mesh stress
test for grid-free topological insertion. It is intentionally separated from
the engineering interpretation of the internal-cooling box: the Bunny tests
geometry complexity, mesh repair, closest-boundary queries, interior candidate
sampling, and insertion on a nontrivial triangle surface.

The canonical zippered reconstruction contains five holes in its bottom. The
script removes unreferenced vertices, consistently orients the mesh, caps the
five boundary loops, normalizes the height to one, and verifies that the result
is a closed edge- and vertex-manifold genus-zero surface before any PDE solve.
The repaired mesh is written to `output/stanford-bunny-watertight.ply`.

## Run

```bash
python -m pip install libigl==2.6.3
make bunny-reference
```

The seeded experiment compares fresh-validation TWoS, scout reuse, a structured
voxel adjoint/TD reference, and random feasible insertion. All reported
compliance values use fresh WoS paths on a common Sobol probe set.

Outputs include raw design trajectories, selected centers, mesh-repair
statistics, a watertight normalized PLY, and PDF/PNG publication figures.

## Data attribution

`assets/bun_zipper.ply` is the canonical high-resolution zippered
reconstruction from the Stanford 3D Scanning Repository:

- Source: Stanford University Computer Graphics Laboratory
- Original archive: `https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz`
- Archive SHA-256: `a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84`
- Mesh: 35,947 stored vertices and 69,451 triangles before removing
  unreferenced vertices

The repository permits research use, free redistribution, and publication of
images in scholarly articles with credit to the Stanford Computer Graphics
Laboratory. The model is not for unrelated commercial product use without
permission. See `https://graphics.stanford.edu/data/3Dscanrep/`.
