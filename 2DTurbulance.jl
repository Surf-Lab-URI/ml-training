using DrWatson
@quickactivate "ml-training"

using Oceananigans
using Statistics
using Printf
using CairoMakie
# using GLMakie
using StructArrays
using JLD2
using DataFrames
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Dates
using CUDA
using ArgParse
using SpecialFunctions

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--jet_amp", "-a"
            help = "amplitude of jet mode in streamfunction"
            arg_type = Float64
            default = 300
        "--n_max", "-n"
            help = "number of modes in streamfunction"
            arg_type = Int
            default = 21
        "--m_jet", "-j"
            help = "wavenumber of horizontal jet"
            arg_type = Int
            default = 2
        "--t_end", "-t"
            help = "end time of simulation"
            arg_type = Float64
            default = nothing
        "--nt"
            help = "number of timesteps recorded"
            arg_type = Int
            default = 20
        "--out_dir", "-o"
            help = "output directory"
            arg_type = String
            default = "out/"
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
tag!(parsed_args)
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

out_dir = parsed_args["out_dir"]
mkpath(out_dir)

# Grid Setup

N = 512
M = 512
grid = RectilinearGrid(GPU(), size=(N, M), extent=(N, M), topology=(Periodic, Periodic, Flat))

# Particle Setup
# Seed one particle at the center of each (x, y) grid cell.

xs = xnodes(grid, Center())
ys = ynodes(grid, Center())

Nx = length(xs)
Ny = length(ys)
# Nparticles = Nx * Ny

# All combinations of cell-center coordinates:
# x varies fastest, y varies slowest.
# Δxₚ = 4
# x₀ = repeat(xs[1:Δxₚ:end], outer = Ny)
# y₀ = repeat(ys[1:Δxₚ:end], inner = Nx)
# Nparticles = length(x₀)
Nparticles = Int(M*N/16)
x₀ = rand(Nparticles)*M
y₀ = rand(Nparticles)*N
z₀ = zeros(Nparticles)

lagrangian_particles = LagrangianParticles(; x = CuArray(x₀), y = CuArray(y₀), z = CuArray(z₀))

# Model Setup

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
                            closure = ScalarDiffusivity(ν=1e-5),
                            particles = lagrangian_particles)

# Random Initial Conditions

u, v, w = model.velocities

A = parsed_args["jet_amp"]*(1.5-rand()) #Amplitude of a long wave added at the end to create jets.
nmax = parsed_args["n_max"]
mmax = parsed_args["n_max"]
mjet = parsed_args["m_jet"]

a = rand(M,N)/(nmax^2)*(21^2) #amplitude of random modes increases if there are fewer of them
k(n) = 2*π*(n-1)/N
l(m) = 2*π*(m-1)/M
ϕ = rand(M,N)*2*π
ϕⱼ = rand()*2*π

# ψ(x,y) = sum(a[m,n]*cos(k(n-11)*x + l(m-11)*y-ϕ[m,n]) for m in 1:21 for n in 1:21)*1e-3 + cos(k(2)*x -ϕ[2,1]) 
ψ(x,y) = A*cos(l(round(mjet*sin(ϕⱼ)))*y + k(round(mjet*cos(ϕⱼ)))*x - ϕ[1,2]) +  sum(a[m,n]*cos(k(n-floor(nmax/2+1))*x + l(m-floor(mmax/2+1))*y-ϕ[m,n]) for m in 1:mmax for n in 1:nmax)
# ψᵢ = ψ.(x,y')
ψf = CenterField(grid)
set!(ψf, ψ)
fill_halo_regions!(ψf)

uᵢ = ∂y(ψf)
compute!(uᵢ)

vᵢ = -∂x(ψf)
compute!(vᵢ)


set!(model, u=uᵢ, v=vᵢ)

# Plotting Initial Conditions

sᵢ = Field(sqrt(u^2 + v^2))
compute!(sᵢ)

s₂ = dropdims(interior(sᵢ); dims=3)


# figᵢ = Figure(size = (800, 500))

# axᵢ = Axis(figᵢ[1, 1], title = "initial velocity (mag)", xlabel = "x", ylabel = "y")
# hmᵢ = heatmap!(axᵢ, s₂, colormap = :viridis)
# # Makie.Colorbar(figᵢ[1,2], hmᵢ, label = "|u|")
# figᵢ
# save("initial_velocity.png", figᵢ)

# Setting Up Simulation

sₘ = maximum(s₂)
tcfl = 0.5*grid.Δxᶠᵃᵃ/sₘ
dt = tcfl*10
if isnothing(parsed_args["t_end"])
    st = parsed_args["nt"]*dt
else
    st = parsed_args["t_end"]
end
simulation = Simulation(model, Δt=tcfl, stop_time=st)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=2*tcfl)        # The TimeStepWizard helps ensure stable time-stepping with a Courant-Freidrichs-Lewy (CFL) number of 0.7.
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

filename = "$(now(UTC))_2DT-A$(A)-nmax$(nmax)-mjet$(mjet)"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, s, div, u, v), #, parsed_args), addition of parsed_args made sim crash
                                                schedule = TimeInterval(dt),
                                                filename = out_dir * filename * ".jld2",
                                                with_halos = false,
                                                overwrite_existing = true)


simulation.output_writers[:particles] = JLD2Writer(model, (; particles = model.particles),
                                                schedule = TimeInterval(dt),
                                                with_halos = false,                      
                                                filename = out_dir*filename * "_particles.jld2",
                                                overwrite_existing = true)

# Running Simulation 

run!(simulation)

# Visualizing Results

ω_timeseries = FieldTimeSeries(out_dir * filename * ".jld2", "ω")
s_timeseries = FieldTimeSeries(out_dir * filename * ".jld2", "s")

times = ω_timeseries.times

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

# Load particle output file
pfile = jldopen(out_dir * filename * "_particles.jld2", "r")

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

hms = heatmap!(ax_s, s; colormap = :speed, colorrange = (0, 5))

Colorbar(fig[2, 2], hmω, label = "ω")
Colorbar(fig[2, 4], hms, label = "s")

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

fullfname = out_dir * filename * ".jld2"
fullfname_particles = out_dir * filename * "_particles.jld2"
combined_name = out_dir * filename * ".npz"
current_dir = pwd()

run(`bash -c "
     source activate base
     conda activate ml-training
     cd $current_dir
     python load_jld2_particles.py $fullfname_particles --fields_path $fullfname --field_a u --field_b v --export_imagegen_npz $combined_name"`)

fig

# Recording Movie

# frames = 1:5:length(times)

# @info "Making animation of vorticity and speed..."

# Makie.record(fig, filename * ".mp4", frames, framerate=24) do i
#     n[] = i
#     x, y = read_xy_at_frame(pts, pkeys, i)
#     px[] = x
#     py[] = y

# end

@info "Done"