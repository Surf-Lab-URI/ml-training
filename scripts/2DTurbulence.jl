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
using TOML   # stdlib — for the per-sample metadata sidecar (§6)

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

# --- IC physical characterization (for §6 metadata) ---
# k_p = √(enstrophy/energy) computed in physical space (no FFT): ω = ∂v/∂x − ∂u/∂y.
U_max_ic = sₘ
energy_ic = enstrophy_ic = k_p_ic = NaN   # defaults if the computation below fails
try
    ω_ic = Field(∂x(v) - ∂y(u)); compute!(ω_ic)
    ω_ic2 = dropdims(interior(ω_ic); dims=3)
    global energy_ic    = 0.5 * mean(s₂.^2)
    global enstrophy_ic = 0.5 * mean(ω_ic2.^2)
    global k_p_ic       = sqrt(mean(ω_ic2.^2) / mean(s₂.^2))
catch err
    @warn "IC physics computation failed (metadata only; run continues)" exception=err
end

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
# Output root is configurable for HPC array jobs: each SLURM task sets PIV_OUT_DIR
# to its own scratch dir so parallel tasks never collide. Defaults to the repo.
data_root = get(ENV, "PIV_OUT_DIR", projectdir())
out_dir = joinpath(data_root, "data", "binary") * "/"
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

# --- Per-sample metadata sidecar (§6) — wrapped so a metadata failure never kills the run ---
try
    uf, vf, wf = model.velocities          # model.velocities now hold the FINAL frame
    s_fin = Field(sqrt(uf^2 + vf^2)); compute!(s_fin)
    s_fin2 = dropdims(interior(s_fin); dims=3)
    ω_fin = Field(∂x(vf) - ∂y(uf)); compute!(ω_fin)
    ω_fin2 = dropdims(interior(ω_fin); dims=3)
    U_max_fin     = maximum(s_fin2)
    energy_fin    = 0.5 * mean(s_fin2.^2)
    enstrophy_fin = 0.5 * mean(ω_fin2.^2)
    k_p_fin       = sqrt(mean(ω_fin2.^2) / mean(s_fin2.^2))
    t_final       = time(simulation)

    git_sha   = try strip(read(`git -C $(projectdir()) rev-parse HEAD`, String)) catch; "unknown" end
    git_dirty = try string(!isempty(strip(read(`git -C $(projectdir()) status --porcelain`, String)))) catch; "unknown" end

    meta = Dict{String,Any}(
        "reproducibility" => Dict{String,Any}(
            "seed"          => parsed_args["seed"],
            "jet_amp_arg"   => parsed_args["jet_amp"],
            "n_max"         => nmax,
            "m_jet"         => mjet,
            "nt"            => nt,
            "stop_time"     => st,
            "julia_version" => string(VERSION),
            "git_sha"       => git_sha,
            "git_dirty"     => git_dirty,
            "grid_N"        => N,
            "grid_M"        => M,
            "extent"        => Float64(N),
            "viscosity_nu"  => 1e-5,
            "advection"     => "WENO(order=5)",
        ),
        "ic_spec" => Dict{String,Any}(
            "streamfunction_form" => "A*cos(l(round(mjet*sin(phij)))*y + k(round(mjet*cos(phij)))*x - phi[1,2]) + sum_{m,n} a[m,n]*cos(k(n-..)*x + l(m-..)*y - phi[m,n])",
            "A_jet_amplitude" => A,
            "phi_jet"         => ϕⱼ,
            "jet_k_index"     => round(mjet*cos(ϕⱼ)),
            "jet_l_index"     => round(mjet*sin(ϕⱼ)),
            "a_mode_max"      => maximum(a),
            "a_mode_mean"     => mean(a),
            "n_modes"         => nmax,
        ),
        "ic_physics" => Dict{String,Any}(
            "U_max"     => U_max_ic,
            "energy"    => energy_ic,
            "enstrophy" => enstrophy_ic,
            "k_p"       => k_p_ic,
            "dt_save"   => dt,
        ),
        "sampling_final" => Dict{String,Any}(
            "t_final"    => t_final,
            "U_max"      => U_max_fin,
            "energy"     => energy_fin,
            "enstrophy"  => enstrophy_fin,
            "k_p"        => k_p_fin,
            "C_achieved" => t_final * U_max_ic * k_p_ic,   # dimensionless sampling age (§3)
        ),
        "particles" => Dict{String,Any}(
            "Nparticles_pool" => Nparticles,
        ),
        "files" => Dict{String,Any}(
            "fields"    => "fields" * vars * ".jld2",
            "particles" => "particles" * vars * ".jld2",
            "combined"  => vars * "_combined.jld2",
        ),
    )
    metafile = out_dir * "metadata" * vars * ".toml"
    open(metafile, "w") do io
        TOML.print(io, meta)
    end
    @info "Wrote metadata sidecar: $metafile"
catch err
    @warn "Metadata write failed (sim output is unaffected)" exception=err
end

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