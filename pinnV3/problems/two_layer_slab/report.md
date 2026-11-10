# Slab with variable conductivity and a disk heat source

- Identifier: `two_layer_slab`
- Family: generic 2D conduction with overlapping regions
- Origin: V3 synthetic verification problem.

The problem contains a rectangular background, an overlapping slab with spatially varying conductivity, and a circular source region. Region order defines ownership: the later region overrides the background. The boundary mix is Dirichlet, Neumann and Robin, with all Robin parameters explicit in the JSON.
