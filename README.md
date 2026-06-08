# Work Flow for ML Training Data Set
2D Turbulence Simulation -> Combine and Conquer -> Image Gen

# Install Julia if you don't have it
Install in your home directory on a super computer cluster like Unity or Expanse using instructions [here](https://julialang.org/downloads/).

# Setup Julia environment
```julia
julia> cd("path/to/project")
pkg> activate .
pkg> instantiate
```

# Automation
The simulation outputs two .jld2 files, one of field data and the other of particle positions. After the simulation has ran, these two outputs will automatically be passed into CombineAndConcquer.jl which combines the outputs into one .jld2 which is used in ImageGen.jl. 

By default, ImageGen.jl will run following CombineAndConquer.jl; however, by passing the `--no_image_gen` flag when running 2DTurbulence.jl:
```bash
julia scripts/2DTurbulence.jl -t 100 --no_image_gen
```
only the simulation and combine and concquer will run. This way ImageGen.jl can be fine tuned to specific pixel displacements.

For reproducible runs, pass `--seed <int>` (default 1234); the same seed and arguments regenerate the same flow and particle seeding:
```bash
julia scripts/2DTurbulence.jl --nt 20 --seed 42
```

Also by default, for storage purposes, pngs of all the image pairs are not produced. The first and last image pairs are shown for debugging purposes. To generate all image pairs, run ImageGen.jl using the -p flag:
```bash
julia scripts/ImageGen.jl -f (out_dir * "combined" * vars * ".jld2") -v (vars) -p
```
This will then generate all image pairs at 10, 15, 20, 25, and 30 pixel displacements.  
