# Work Flow for ML Training Data Set
2D Turbulence Simulation -> Combine and Conquer -> Image Gen

# Automation
The simulation outputs two .jld2 files, one of field data and the other of particle positions. After the simulation has ran, these two outputs will automatically be passed into CombineAndConcquer.jl which combines the outputs into one .jld2 which is used in ImageGen.jl. 

By default, ImageGen.jl will run following CombineAndConquer.jl; however, by including the argument:
```bash
-m false
```
only the simulation and combine and concquer will run. This way ImageGen.jl can be fine tuned to specific pixel displacements. 

# Install Julia if you don't have it
Install in your home directory on a super computer cluster like Unity or Expanse using instructions [here](https://julialang.org/downloads/).

# Setup Julia environment
```julia
julia> cd("path/to/project")
pkg> activate .
pkg> instantiate
```

