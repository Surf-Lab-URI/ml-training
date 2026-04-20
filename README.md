work flow for ml training data set:
2D Turbulence Simulation (julia) -> load jld2 particles (python) -> image gen (python)

# Setup Conda Environment
create a conda environment from environment.yml
```bash
conda env create -f environment.yml
```

# Install Julia if you don't have it
Install in your home directory on a super computer cluster like Unity or Expanse using instructions [here](https://julialang.org/downloads/).

# Setup Julia environment
```julia
julia> cd("path/to/project")
pkg> activate .
pkg> instantiate
```

