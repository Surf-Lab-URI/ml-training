using Oceananigans
using Statistics
using Printf
using CairoMakie
using StructArrays
using JLD2
using DataFrames

# Grid Setup

grid = RectilinearGrid(size=(256, 256), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

# Particle Setup
# Seed one particle at the center of each (x, y) grid cell.

xs = xnodes(grid, Center())
ys = ynodes(grid, Center())

Nx = length(xs)
Ny = length(ys)
Nparticles = Nx * Ny

# All combinations of cell-center coordinates:
# x varies fastest, y varies slowest.
x₀ = repeat(xs, outer = Ny)
y₀ = repeat(ys, inner = Nx)
z₀ = zeros(Nparticles)

lagrangian_particles = LagrangianParticles(; x = x₀, y = y₀, z = z₀)

# Model Setup

model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=1e-5),
                            particles = lagrangian_particles)

# Random Initial Conditions

u, v, w = model.velocities

uᵢ = 4ones((size(u)...))
vᵢ = 3ones((size(v)...))

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

# Plotting Initial Conditions

sᵢ = Field(sqrt(u^2 + v^2))
compute!(sᵢ)

s₂ = dropdims(interior(sᵢ); dims=3)

figᵢ = Figure(size = (800, 500))

axᵢ = Axis(figᵢ[1, 1], title = "initial velocity (mag)", xlabel = "x", ylabel = "y")
hmᵢ = heatmap!(axᵢ, s₂, colormap = :viridis)
Colorbar(figᵢ[1,2], hmᵢ, label = "|u|")

save("initial_velocity.png", figᵢ)

# Setting Up Simulation

simulation = Simulation(model, Δt=0.1, stop_time=50)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)        # The TimeStepWizard helps ensure stable time-stepping with a Courant-Freidrichs-Lewy (CFL) number of 0.7.
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Logging Simulation Progress

function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s\n",
                            iteration(sim), time(sim), sim.Δt, max_abs_u, walltime)
end

# Computing Vorticity and Speed

u, v, w = model.velocities

ω = ∂x(v) - ∂y(u)

div = ∂x(u) + ∂y(v)

s = sqrt(u^2 + v^2)

filename = "2D_Turbulance(particles)"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, s, div, u ,v),
                                                schedule = TimeInterval(0.2),
                                                filename = filename * ".jld2",
                                                overwrite_existing = true)

simulation.output_writers[:particles] = JLD2Writer(model, (; particles = model.particles),
                                                schedule = TimeInterval(0.2),                      
                                                filename = filename * "_particles.jld2",
                                                overwrite_existing = true)

# Running Simulation 

run!(simulation)

# Visualizing Results

ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")

times = ω_timeseries.times

set_theme!(Theme(fontsize = 20))

fig = Figure(size = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               limits = ((0, 2π), (0, 2π)),
               aspect = AxisAspect(1))

ax_ω = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 3]; title = "Speed", axis_kwargs...)
xlims!(ax_ω, minimum(xnodes(grid, Center())), maximum(xnodes(grid, Center())))
ylims!(ax_ω, minimum(ynodes(grid, Center())), maximum(ynodes(grid, Center())))

# Load particle output file
pfile = jldopen(filename * "_particles.jld2", "r")

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

# Plotting

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

hms = heatmap!(ax_s, s; colormap = :speed, colorrange = (0, 0.2))

Colorbar(fig[2, 2], hmω, label = "ω")
Colorbar(fig[2, 4], hms, label = "s")

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

fig

# Recording Movie

frames = 1:length(times)

@info "Making animation of vorticity and speed..."

#record(fig, filename * ".mp4", frames, framerate=24) do i
    #n[] = i
    #x, y = read_xy_at_frame(pts, pkeys, i)
    #px[] = x
    #py[] = y

#end

@info "Done"