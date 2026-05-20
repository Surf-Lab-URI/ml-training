using DrWatson
@quickactivate "ml-training"

using JLD2
using ArgParse
using Oceananigans
using Images
using FileIO
using ImageIO
using Printf
using Random
using Statistics
using Dates
using CUDA
using SpecialFunctions

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--combined_file", "-f"
            help = "path to combined JLD2 file of fields and particles"
            default = nothing
        "--input_dir", "-d"
            help = "path where combined simulation output JLD2 files are stored"
            default = nothing
        "--vars", "-v"
            help = "variables used in original simulation"
            arg_type = String
            default = nothing
        "--name", "-n"
            help = "base name for output file"
            arg_type = String
            default = nothing
        "--save_pngs", "-p"
            help = "save image pairs as PNG files"
            action = :store_true
        "--sample", "-k"
            help = "number of particles to render in each image pair"
            arg_type = Int
            default = 5000
        "--seed", "-s"
            help = "random seed used for particle subsampling"
            arg_type = Int
            default = 1234
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

file = parsed_args["combined_file"]
input_dir = parsed_args["input_dir"]
vars = parsed_args["vars"]
name = parsed_args["name"]
save_pngs = parsed_args["save_pngs"]
k_particles = parsed_args["sample"]
seed = parsed_args["seed"]
rng = MersenneTwister(seed)
out_dir = projectdir() * "/data/visual/"

# ---Opening Combined JLD2 File---

function get_frame_keys(file)
    keys_raw = collect(keys(file["fields/timeseries/u"]))
    frame_keys = filter(k -> k != "serialized", keys_raw)
    sort!(frame_keys, by = k -> parse(Int, k))
    return frame_keys
end

function load_particle_frame(file, frame_key)
    p = file["particles/timeseries/particles/$frame_key"]

    x = Array(p.x)
    y = Array(p.y)

    return x, y
end

function load_field_frame(file, frame_key)
    u = Array(file["fields/timeseries/u/$frame_key"])
    v = Array(file["fields/timeseries/v/$frame_key"])

    if ndims(u) == 3 && size(u, 3) == 1
        u = dropdims(u, dims = 3)
    end

    if ndims(v) == 3 && size(v, 3) == 1
        v = dropdims(v, dims = 3)
    end

    return u, v
end

function load_time(file, frame_key)
    return file["fields/timeseries/t/$frame_key"]
end

# ---Making Image Pair at Each Frame---

function render_particles(x, y;
    width = 512,
    height = 512,
    xlim = (0.0, 2π),
    ylim = (0.0, 2π),
    background = 0.0,
    peak = 1.0,
    σₚ = 1.2)

    # blank image as an array
    img = fill(Float32(background), height, width)

    # boundries of the image
    xmin, xmax = xlim
    ymin, ymax = ylim

    # wrap particle positions into the periodic domain instead of clamping them to the edges
    Lx = xmax - xmin
    Ly = ymax - ymin

    x_wrapped = xmin .+ mod.(x .- xmin, Lx)
    y_wrapped = ymin .+ mod.(y .- ymin, Ly)

    # convert particle positions to pixel coordinates
    u = (x_wrapped .- xmin) ./ Lx .* (width - 1)
    v = (y_wrapped .- ymin) ./ Ly .* (height - 1)

    # Gaussian kernels
    radius = max(1, Int(ceil(3σₚ)))
    grid = collect(-radius:radius)

    g = exp.(-(grid.^2) ./ (2σₚ^2))

    kernel = g * g'
    kernel ./= maximum(kernel)
    kernel .*= peak


    # render each particle as a Gaussian blob
    for (ui, vi) in zip(u, v)
        cx = round(Int, ui)
        cy = round(Int, vi)

        # image coordinates 
        x₀ = cx - radius + 1
        y₀ = cy - radius + 1

        x₁ = cx + radius + 1
        y₁ = cy + radius + 1

        # clipping to image bounds
        x₀_clipped = max(1, x₀)
        y₀_clipped = max(1, y₀)

        x₁_clipped = min(width, x₁)
        y₁_clipped = min(height, y₁)

        kx₀ = x₀_clipped - x₀ + 1
        ky₀ = y₀_clipped - y₀ + 1

        kx₁ = kx₀ + (x₁_clipped - x₀_clipped)
        ky₁ = ky₀ + (y₁_clipped - y₀_clipped)

        # add the kernel to the image
        img[y₀_clipped:y₁_clipped, x₀_clipped:x₁_clipped] .+= kernel[ky₀:ky₁, kx₀:kx₁]
    end
    return img
end

# ---Converting Point Image to UInt8---

function to_uint8(img; clip_max = 3.0)

    img = clamp.(img, 0.0, clip_max) ./ clip_max

    return UInt8.(round.(img .* 255))
end

# ---Fitting Field Data to Image Size---

function fit_field_2d(arry, Ny, Nx)

    # reshape to 2D
    a = Array(arry)

    if ndims(a) == 3 && size(a, 3) == 1
        a = dropdims(a, dims=3)
    end

    iNy, iNx = size(a)

    # crop if too large
    if iNy > Ny
        a = a[1:Ny, :]
    end

    if iNx > Nx
        a = a[:, 1:Nx]
    end

    iNy, iNx = size(a)

    # pad if too small
    if iNy < Ny
        pad = zeros(eltype(a), Ny - iNy, size(a, 2))
        a = vcat(a, pad)
    end

    if iNx < Nx
        pad = zeros(eltype(a), size(a, 1), Nx - iNx)
        a = hcat(a, pad)
    end

    return a
end

# ---Saving Image as a JLD2 File---

function save_image(jldfile, imgA, imgB, pair_index)
    pair_key = @sprintf("pairs/%06d", pair_index)

    jldfile["$pair_key/A"] = to_uint8(imgA)
    jldfile["$pair_key/B"] = to_uint8(imgB)

    return nothing
end

# ---Loading u and v fields---

function save_field_pairs(jldfile, uA, vA, uB, vB, pair_index;
    Δt_pair = 1.0,
    Ny = 512,
    Nx = 512)

    pair_key = @sprintf("pairs/%06d", pair_index)

    uA = fit_field_2d(uA, Ny, Nx)
    vA = fit_field_2d(vA, Ny, Nx)
    uB = fit_field_2d(uB, Ny, Nx)
    vB = fit_field_2d(vB, Ny, Nx)

    jldfile["$pair_key/fields/uA"] = Float32.(uA .* Δt_pair)
    jldfile["$pair_key/fields/vA"] = Float32.(vA .* Δt_pair)
    jldfile["$pair_key/fields/uB"] = Float32.(uB .* Δt_pair)
    jldfile["$pair_key/fields/vB"] = Float32.(vB .* Δt_pair)

    return nothing
end

# ---Selecting a Subset of Particles---

function subset_particles(xA, yA, xB, yB, k, rng)
    nparticles = min(length(xA), length(yA), length(xB), length(yB))
    k_use = min(k, nparticles)

    idx = randperm(rng, nparticles)[1:k_use]

    return xA[idx], yA[idx], xB[idx], yB[idx]
end

# ---Making Imgage Pairs---

function make_image_pair(jldfile, xA, yA, xB, yB, uA, vA, uB, vB, pair_index;
        width = 512,
        height = 512,
        xlim = (0.0, 2π),
        ylim = (0.0, 2π),
        σₚ = 1.2,
        Δt_pair = 1.0)

    imgA = render_particles(xA, yA; width, height, xlim, ylim, σₚ)
    imgB = render_particles(xB, yB; width, height, xlim, ylim, σₚ)

    save_image(jldfile, imgA, imgB, pair_index)

    save_field_pairs(
        jldfile,
        uA, vA,
        uB, vB,
        pair_index;
        Δt_pair = Δt_pair,
        Ny = height,
        Nx = width
    )

    return imgA, imgB
end

function save_image_png(imgA, imgB, pair_index; out_dir = ".", name = "pair")
    mkpath(out_dir)

    fileA = joinpath(out_dir, @sprintf("%s_%06d_a.png", name, pair_index))
    fileB = joinpath(out_dir, @sprintf("%s_%06d_b.png", name, pair_index))

    save(fileA, Gray.(to_uint8(imgA) ./ 255))
    save(fileB, Gray.(to_uint8(imgB) ./ 255))

    return fileA, fileB
end
