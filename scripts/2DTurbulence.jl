using DrWatson
@quickactivate "ml-training"

include(projectdir()*"/src/args.jl")

using Oceananigans
using Statistics
using Printf
using StructArrays
using JLD2
using Random
using DataFrames
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Dates
using CUDA
using SpecialFunctions

# ---Reproducibility: one master seed drives every simulation rand() (BUG-8)---
Random.seed!(parsed_args["seed"])
A = parsed_args["jet_amp"]*(1.5-rand())   # moved here from args.jl so the seed applies (BUG-8)

# ---Grid Setup---

N = 512
M = 512
#grid = RectilinearGrid(GPU(), size=(N, M), extent=(N, M), topology=(Periodic, Periodic, Flat))
grid = RectilinearGrid(size=(N, M), extent=(N, M), topology=(Periodic, Periodic, Flat))

# ---Particle Setup---

Nparticles = Int(M*N/16)
x₀ = rand(Nparticles)*M
y₀ = rand(Nparticles)*N
z₀ = zeros(Nparticles)

#lagrangian_particles = LagrangianParticles(; x = CuArray(x₀), y = CuArray(y₀), z = CuArray(z₀))
lagrangian_particles = LagrangianParticles(; x = x₀, y = y₀, z = z₀)

# ---Model Setup---

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
                            closure = ScalarDiffusivity(ν=1e-5),
                            particles = lagrangian_particles)

# ---Random Initial Conditions---

u, v, w = model.velocities

a = rand(M,N)/(nmax^2)*(21^2) # amplitude of random modes increases if there are fewer of them
k(n) = 2*π*(n-1)/N
l(m) = 2*π*(m-1)/M
ϕ = rand(M,N)*2*π
ϕⱼ = rand()*2*π

# defining stream function
ψ(x,y) = A*cos(l(round(mjet*sin(ϕⱼ)))*y + k(round(mjet*cos(ϕⱼ)))*x - ϕ[1,2]) +  sum(a[m,n]*cos(k(n-floor(nmax/2+1))*x + l(m-floor(mmax/2+1))*y-ϕ[m,n]) for m in 1:mmax for n in 1:nmax)

# setting stream funciton on grid
ψf = CenterField(grid)
set!(ψf, ψ)
fill_halo_regions!(ψf)

# computing velocities from stream function
uᵢ = ∂y(ψf)
vᵢ = -∂x(ψf)
compute!(uᵢ)
compute!(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

# computing speed
sᵢ = Field(sqrt(u^2 + v^2))
compute!(sᵢ)

s₂ = dropdims(interior(sᵢ); dims=3)

# ---Setting Up Simulation---

# defining a stable time step based on the CFL condition
sₘ = maximum(s₂)
tcfl = 0.5*grid.Δxᶠᵃᵃ/sₘ
dt = tcfl*10
st = isnothing(t_end) ? nt*dt : t_end    # moved from args.jl — see BUG-1

simulation = Simulation(model, Δt=tcfl, stop_time=st)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=2*tcfl)        # The TimeStepWizard helps ensure stable time-stepping with a Courant-Freidrichs-Lewy (CFL) number of 0.7.
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ---Logging Simulation Progress---

function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    walltime = prettytime(sim.run_wall_time)

    @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s\n",
                            iteration(sim), time(sim), sim.Δt, max_abs_u, walltime)
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))   # BUG-4: register progress logging

# ---Velocities (the only fields we store; ω, s, div are derived and recomputed on demand)---

u, v, w = model.velocities

# ---Simulation Output Writers---
out_dir = projectdir() * "/data/binary/"
mkpath(out_dir)
vars = "_$(now(UTC))_2DT-A$(A)-nmax$(nmax)-mjet$(mjet)"

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v),   # only velocities; ω/s/div are derived (recompute from u,v when needed)
                                                schedule = TimeInterval(dt),
                                                filename = out_dir * "fields" * vars * ".jld2",
                                                with_halos = false,
                                                overwrite_existing = true)


simulation.output_writers[:particles] = JLD2Writer(model, (; particles = model.particles),
                                                schedule = TimeInterval(dt),
                                                with_halos = false,                      
                                                filename = out_dir * "particles" * vars * ".jld2",
                                                overwrite_existing = true)

# ---Running Simulation---

run!(simulation)

@info "Simulation complete. Now combining output files..."

# --- Combining JLD2 Output Files ---

run(`$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir() * "/src/CombineAndConquer.jl") -f $(out_dir * "fields" * vars * ".jld2") -p $(out_dir * "particles" * vars * ".jld2") -s $(vars)`)
@info "Output files combined. "
@info "New file located at: $(out_dir * vars * "_combined.jld2")"

# --- Generating Image Pairs ---

if automate == true
    @info "Generating image pairs..."
    run(`$(Base.julia_cmd()) --project=$(projectdir()) $(projectdir() * "/scripts/ImageGen.jl") -f $(out_dir * vars * "_combined.jld2") -v $(vars) -s $(parsed_args["seed"])`)
else
    @info "Skipping image pair generation."
end

@info "Done"