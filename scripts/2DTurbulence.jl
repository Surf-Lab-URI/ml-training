using DrWatson
@quickactivate "ml-training"

include(projectdir()*"/src/args.jl")

using Oceananigans
using Statistics
using Printf
using CairoMakie
using StructArrays
using JLD2
using DataFrames
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Dates
using CUDA
using SpecialFunctions



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

simulation = Simulation(model, Δt=tcfl, stop_time=st)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=2*tcfl)        # The TimeStepWizard helps ensure stable time-stepping with a Courant-Freidrichs-Lewy (CFL) number of 0.7.
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ---Logging Simulation Progress---

function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s\n",
                            iteration(sim), time(sim), sim.Δt, max_abs_u, walltime)
end

# ---Computing Vorticity and Speed---

u, v, w = model.velocities

ω = ∂x(v) - ∂y(u)

div = ∂x(u) + ∂y(v)

s = sqrt(u^2 + v^2)

# ---Simulation Output Writers---
out_dir = projectdir() * "/data/binary/"
vars = "_$(now(UTC))_2DT-A$(A)-nmax$(nmax)-mjet$(mjet)"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, s, div, u, v), #, parsed_args), addition of parsed_args made sim crash
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

# ---Visualizing Results---

# loading vorticity and speed timeseries from JLD2 files
ω_timeseries = FieldTimeSeries(out_dir * "fields" * vars * ".jld2", "ω")
s_timeseries = FieldTimeSeries(out_dir * "fields" * vars * ".jld2", "s")

times = ω_timeseries.times

# loading particle output file
pfile = jldopen(out_dir * "particles" * vars * ".jld2", "r")

ts = pfile["timeseries"]
pts = ts["particles"]

# pulling particle keys -> flitering -> sorting
raw_keys = collect(keys(pts))
pkeys = filter(k -> all(isdigit, k), raw_keys)
sort!(pkeys, by = k -> parse(Int, k))

function read_xy_at_frame(pts, pkeys, i)
    k = pkeys[clamp(i, 1, length(pkeys))]     
    snap = pts[k]                             
    if hasproperty(snap, :x) && hasproperty(snap, :y)
        return getproperty(snap, :x), getproperty(snap, :y)
    end

    if hasproperty(snap, :particles)
        p = getproperty(snap, :particles)
        if hasproperty(p, :x) && hasproperty(p, :y)
            return getproperty(p, :x), getproperty(p, :y)
        end
    end
    
end

n = Observable(1)

# animating results with Makie
set_theme!(Theme(fontsize = 20))

fig = Figure(size = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               limits = ((0, N), (0, M)),
               aspect = AxisAspect(1))

ax_ω = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 3]; title = "Speed", axis_kwargs...)

xlims!(ax_ω, minimum(xnodes(grid, Center())), maximum(xnodes(grid, Center())))
ylims!(ax_ω, minimum(ynodes(grid, Center())), maximum(ynodes(grid, Center())))

# plotting vorticity and speed
ω = @lift ω_timeseries[$n]
s = @lift s_timeseries[$n]

hmω = heatmap!(ax_ω, ω; colormap = :balance, colorrange = (-2, 2))

px = Observable(Float64[])
py = Observable(Float64[])

scatter!(ax_ω, px, py;
    markersize = 1,
    strokewidth = 0.1,
    color = :green,
    strokecolor = :green)

hms = heatmap!(ax_s, s; colormap = :speed, colorrange = (0, 5))

Colorbar(fig[2, 2], hmω, label = "ω")
Colorbar(fig[2, 4], hms, label = "s")

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

# Recording Movie
# frames = 1:5:length(times)

# @info "Making animation of vorticity and speed..."

# Makie.record(fig, filename * ".mp4", frames, framerate=24) do i
#     n[] = i
#     x, y = read_xy_at_frame(pts, pkeys, i)
#     px[] = x
#     py[] = y

@info "Done"