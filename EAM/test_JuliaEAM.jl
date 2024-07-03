# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Julia 1.10.4
#     language: julia
#     name: julia-1.10
# ---

# +
cd(@__DIR__)
ENV["CELLLISTMAP_8.3_WARNING"] = "false"

using Pkg
Pkg.activate(".")

using Printf
using AtomsCalculators
using ASEconvert # use this PR:https://github.com/mfherbst/ASEconvert.jl/pull/17, Pkg.add(url="https://github.com/tjjarvinen/ASEconvert.jl.git", rev="atomscalculators")
using Unitful: Å, nm
using PythonCall
ENV["PYTHON"] = "/SNS/users/ccu/miniconda3/envs/analysis/bin/python"
# install the following packages in julia REPL
# using CondaPkg
# CondaPkg.add_pip("IPython")
# CondaPkg.add_pip("nglview")
using StaticArrays: SVector

using GLMakie
using Molly
using Zygote
using LinearAlgebra
import Interpolations:CubicSplineInterpolation, interpolate, BSpline, Cubic, scale, Line, OnGrid, extrapolate, Gridded
using DelimitedFiles
# -

function repeat(fun,times)
    for i in 1:times
        fun()
    end
end

# ## ASE EAM for reference

# +
## 1. Import ASE and other Python modules
# Import ASE and other Python modules as needed
ase = pyimport("ase")
ase_view = pyimport("ase.visualize")
ase_plot = pyimport("ase.visualize.plot")
plt = pyimport("matplotlib.pyplot")

# Import Al EAM potential
fname = "Al99.eam.alloy"
EAM_ASE = pyimport("ase.calculators.eam") # python ASE-EAM calculator
eam_cal = ASEconvert.ASEcalculator(EAM_ASE.EAM(potential=fname))  # EAM calculater, converted to AtomsBase format

## 2. Define customized interaction type in AtomsCalculators
### 2.1 Define interaction
# AtomsCalculators class containing calculator and system
struct EAMInteraction
    calculator::Any  # Holds the ASE EAM calculator reference
    atoms_ab::Any    # Holds atoms representation compatible with ASE
end

### 2.2 Customized convert_ase function for evaluating potential and interactions using the ASE EAM interaction
# Customized convert_ase function converting Molly system to ASE format: handling with charges
using UnitfulAtomic
import PeriodicTable
const uVelocity = sqrt(u"eV" / u"u")
function convert_ase_custom(system::AbstractSystem{D}) where {D}
    # print("called by Molly")
    D != 3 && @warn "1D and 2D systems not yet fully supported."

    n_atoms = length(system)
    pbc     = map(isequal(Periodic()), boundary_conditions(system))
    numbers = atomic_number(system)
    masses  = ustrip.(u"u", atomic_mass(system))

    symbols_match = [
        PeriodicTable.elements[atnum].symbol == string(atomic_symbol(system, i))
        for (i, atnum) in enumerate(numbers)
    ]
    if !all(symbols_match)
        @warn("Mismatch between atomic numbers and atomic symbols, which is not " *
              "supported in ASE. Atomic numbers take preference.")
    end

    cell = zeros(3, 3)
    for (i, v) in enumerate(bounding_box(system))
        cell[i, 1:D] = ustrip.(u"Å", v)
    end

    positions = zeros(n_atoms, 3)
    for at = 1:n_atoms
        positions[at, 1:D] = ustrip.(u"Å", position(system, at))
    end

    velocities = nothing
    if !ismissing(velocity(system))
        velocities = zeros(n_atoms, 3)
        for at = 1:n_atoms
            velocities[at, 1:D] = ustrip.(uVelocity, velocity(system, at))
        end
    end

    # We don't map any extra atom properties, which are not available in ASE as this
    # only causes a mess: ASE could do something to the atoms, but not taking
    # care of the extra properties, thus rendering the extra properties invalid
    # without the user noticing.
    charges = nothing
    magmoms = nothing
    for key in atomkeys(system)
        if key in (:position, :velocity, :atomic_symbol, :atomic_number, :atomic_mass)
            continue  # Already dealt with
        elseif key == :charge
            charges = charge.(system.atoms) #### Using the charge() function in Molly!
        elseif key == :magnetic_moment
            magmoms = system[:, :magnetic_moment]
        else
            @warn "Skipping atomic property $key, which is not supported in ASE."
        end
    end

    # Map extra system properties
    info = Dict{String, Any}()
    for (k, v) in pairs(system)
        if k in (:bounding_box, :boundary_conditions)
            continue
        elseif k in (:charge, )
            info[string(k)] = ustrip(u"e_au", v)
        elseif v isa Quantity || (v isa AbstractArray && eltype(v) <: Quantity)
            # @warn("Unitful quantities are not yet supported in convert_ase. " *
            #       "Ignoring key $k")
        else
            info[string(k)] = v
        end
    end

    ase.Atoms(; positions, numbers, masses, magmoms, charges,
              cell, pbc, velocities, info)
end

### 2.3 Force calculation
# Define customized AtomsCalculators here
function AtomsCalculators.potential_energy(system::Molly.System, interaction::EAMInteraction; kwargs...)
    # Convert Molly's system to ASE's Atoms format
    ase_atoms = convert_ase_custom(system)
    
    # Calculate potential energy using ASE's EAM calculator
    # energy = AtomsCalculators.potential_energy(ase_atoms, interaction.calculator)
    energy_py = interaction.calculator.ase_python_calculator.get_potential_energy(ase_atoms)
    energy = pyconvert(Float64, energy_py)*u"eV" # also consider unit conversion

    return energy
end

function AtomsCalculators.forces(system::Molly.System, interaction::EAMInteraction; kwargs...)
    # Convert Molly's system to ASE's Atoms format
    ase_atoms = convert_ase_custom(system)

    # Use ASE to calculate forces
    f = interaction.calculator.ase_python_calculator.get_forces(ase_atoms)

    # Reshape and rearrange into the jupyter SVector format
    tmp = pyconvert(Array{Float64}, f)
    vector_svector = [SVector{3}(tmp[i, j] for j in 1:3) for i in 1:size(tmp, 1)]
    FT = AtomsCalculators.promote_force_type(system, interaction.calculator.ase_python_calculator)
    tmp2 = [SVector{3}(tmp[i, j] for j in 1:3) for i in 1:size(tmp, 1)]
    tmp3 = reinterpret(FT, tmp2)

    return tmp3
end
# -

# ## Define a Molly system wo interaction 

# +
## 1. Import ASE and other Python modules
# Import ASE and other Python modules as needed
ase = pyimport("ase")

al_LatConst = 4.0495
atom_mass = 26.9815u"u"  # Atomic mass of aluminum in grams per mole
function system_adatom(size)

    # Build an (001) Al surface  
    atoms_ase = ase.build.fcc100("Al", size=size, vacuum = al_LatConst*4)
    # The basis vectors on x and y are along 1/2<110> directions
    ase.build.add_adsorbate(atoms_ase, "Al", al_LatConst/2, position=(al_LatConst*(2.5*sqrt(1/2)/2),al_LatConst*(2.5*sqrt(1/2)/2)))
    # ase.build.add_adsorbate(atoms_ase, "Al", al_LatConst/2, "bridge")

    atoms_ase.translate([al_LatConst*(sqrt(1/2)/4),al_LatConst*(sqrt(1/2)/4),0])
    atoms_ase.wrap()

    atoms_ase_cell = atoms_ase.get_cell()
    box_size = pyconvert(Array{Float64}, [atoms_ase_cell[x,x] for x in range(0,2)])/10*u"nm"

    # Build an Julia AtomsBase abstract 
    atoms_ab = pyconvert(AbstractSystem, atoms_ase)

    ## 4. Create Molly system
    ### 4.1 Convert atom positions to Molly's expected format (nanometers) and create Molly.Atom objects
    # Get atom positions from previously defined ASE system
    function get_positions(atoms_ase)
        positions = [(atom.position[1], atom.position[2], atom.position[3]) for atom in atoms_ase]
        return positions
    end

    # Convert each position from Ångströms to nanometers and ensure the conversion is applied element-wise.
    atom_positions = [SVector(uconvert(nm, pos[1]), 
        uconvert(nm, pos[2]), uconvert(nm, pos[3])) for pos in get_positions(atoms_ab)]

    molly_atoms = [Molly.Atom(index=i, charge=0, mass=atom_mass, 
                            #   σ=2.0u"Å" |> x -> uconvert(u"nm", x), ϵ=ϵ_kJ_per_mol
                            ) for i in 1:length(atom_positions)]
    return molly_atoms, atoms_ab, box_size, atom_positions
end

molly_atoms, atoms_ab, box_size, atom_positions = system_adatom((20,20,24))


# Specify boundary condition
boundary_condition = Molly.CubicBoundary(box_size[1],box_size[2],box_size[3])

# Create the Molly System with atoms, positions, velocities, and boundary
molly_system = Molly.System(
    atoms=molly_atoms,
    atoms_data = [AtomData(element="Al") for a in molly_atoms],
    coords=atom_positions,  # Ensure these are SVector with correct units
    boundary=boundary_condition,
    neighbor_finder = DistanceNeighborFinder(
    eligible=trues(length(molly_atoms), length(molly_atoms)),
    n_steps=10,
    dist_cutoff=6.3/10*u"nm"),
    energy_units=u"eV",  # Ensure these units are correctly specified
    force_units=u"eV/nm"  # Ensure these units are correctly specified
    )
# -

# ## Define interaction

# +
function deriv(spline)
    d_spline(x) = gradient(spline, x)[1]
    return d_spline
end

# Helper function to get the type of a CubicSplineInterpolation object
function get_spline_type()
    dummy_spline = CubicSplineInterpolation(0.0:1.0:2.0, [0.0, 1.0, 2.0])
    return typeof(dummy_spline)
end

# Helper function to get the type of the derivative of a CubicSplineInterpolation object
function get_deriv_type()
    dummy_spline = CubicSplineInterpolation(0.0:1.0:2.0, [0.0, 1.0, 2.0])
    dummy_deriv = deriv(dummy_spline)
    return typeof(dummy_deriv)
end

mutable struct EAM
    Nelements::Int
    elements::Vector{String}
    nrho::Int
    drho::Float64
    nr::Int
    dr::Float64
    cutoff::Float64
    embedded_data::Matrix{Float64}
    density_data::Matrix{Float64}
    Z::Vector{Int}
    mass::Vector{Float64}
    a::Vector{Float64}
    lattice::String
    rphi_data::Array{Float64, 3}
    r_range::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    rho_range::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    r::Vector{Float64}
    rho::Vector{Float64}

    embedded_energy::Vector{get_spline_type()}
    electron_density::Vector{get_spline_type()}
    d_embedded_energy::Vector{get_spline_type()}
    d_electron_density::Vector{get_spline_type()}
    phi::Matrix{get_spline_type()}
    d_phi::Matrix{get_spline_type()}

    EAM() = new()
end

# +
function set_splines(calculator::EAM)
    calculator.embedded_energy = Vector{get_spline_type()}(undef, calculator.Nelements)
    calculator.electron_density = Vector{get_spline_type()}(undef, calculator.Nelements)
    calculator.d_embedded_energy = Vector{get_spline_type()}(undef, calculator.Nelements)
    calculator.d_electron_density = Vector{get_spline_type()}(undef, calculator.Nelements)

    for i in 1:calculator.Nelements
        calculator.embedded_energy[i] = CubicSplineInterpolation(calculator.rho_range, calculator.embedded_data[i, :]) # arrays of embedded energy functions, [N_types]
        calculator.electron_density[i] = CubicSplineInterpolation(calculator.r_range, calculator.density_data[i, :]) # arrays of electron density functions, [N_types]
        f_embedded_energy = deriv(calculator.embedded_energy[i])
        f_electron_density = deriv(calculator.electron_density[i])
        d_embedded_energy_knot = f_embedded_energy.(calculator.rho)
        d_electron_density_knot = f_electron_density.(calculator.r)
        calculator.d_embedded_energy[i] = CubicSplineInterpolation(calculator.rho_range,d_embedded_energy_knot) # arrays of derivative of embedded energy functions, [N_types]
        calculator.d_electron_density[i] = CubicSplineInterpolation(calculator.r_range,d_electron_density_knot) # arrays of derivative of electron density functions, [N_types]
        # calculator.d_embedded_energy[i] = deriv(calculator.embedded_energy[i]) # arrays of derivative of embedded energy functions, [N_types]
        # calculator.d_electron_density[i] = deriv(calculator.electron_density[i]) # arrays of derivative of electron density functions, [N_types]
    end

    calculator.phi = Matrix{get_spline_type()}(undef, calculator.Nelements, calculator.Nelements) # arrays of pairwise energy functions, [N_types, N_types]
    calculator.d_phi = Matrix{get_spline_type()}(undef, calculator.Nelements, calculator.Nelements) # arrays, [N_types, N_types]

    # ignore the first point of the phi data because it is forced
    # to go through zero due to the r*phi format in alloy and adp
    for i in 1:calculator.Nelements
        for j in i:calculator.Nelements
            calculator.phi[i, j] = CubicSplineInterpolation(calculator.r_range[2:end], calculator.rphi_data[i, j, :][2:end] ./ calculator.r[2:end]) 
            f_d_phi = deriv(calculator.phi[i, j])
            d_phi_knot = f_d_phi.(calculator.r[2:end])
            calculator.d_phi[i, j] = CubicSplineInterpolation(calculator.r_range[2:end],d_phi_knot)
            # calculator.d_phi[i, j] = deriv(calculator.phi[i, j])

            if j != i
                calculator.phi[j, i] = calculator.phi[i, j]
                calculator.d_phi[j, i] = calculator.d_phi[i, j]
            end
        end
    end
end


"""
    read_potential!(calculator::EAM, fd::String)

Reads the potential data from a file and populates the fields of the `calculator` object.

# Arguments
- `calculator::EAM`: The EAM calculator object to populate with potential data.
- `fd::String`: The file path to the potential data file.

# Description
This function reads the potential data from the specified file and assigns the values to the corresponding fields of the `calculator` object. The file should be in a specific format, with each line containing the relevant data for a specific field.

The function starts reading the file from the 4th line and converts the lines into a list of strings. It then extracts the number of elements, element symbols, and other parameters from the list. Next, it reads the embedded energy and electron density data for each element, as well as the r*phi data for each interaction between elements. Finally, it sets up the ranges and arrays for the potential data and calls the `set_splines` function to calculate the splines.

Note: This function assumes that the potential data file is formatted correctly and contains the required information in the expected order.

"""
function read_potential!(calculator::EAM, fd::String)
    lines = readdlm(fd, '\n', String) # read the files, split by new line

    function lines_to_list(lines) # convert the entries in lines to list
        data = []
        for line in lines
            append!(data, split(line))
        end
        return data
    end

    i = 4 # start from the 4th line
    data = lines_to_list(lines[i:end])

    calculator.Nelements = parse(Int, data[1]) # number of elements
    d = 2
    calculator.elements = data[d:d+calculator.Nelements-1] # the elements symbols starts from the 2nd entries
    d += calculator.Nelements

    calculator.nrho = parse(Int, data[d]) 
    calculator.drho = parse(Float64, data[d+1])
    calculator.nr = parse(Int, data[d+2])
    calculator.dr = parse(Float64, data[d+3])# the cutoff radius in angstroms
    calculator.cutoff = parse(Float64, data[d+4]) 

    calculator.embedded_data = zeros(calculator.Nelements, calculator.nrho)
    calculator.density_data = zeros(calculator.Nelements, calculator.nr)
    calculator.Z = zeros(Int, calculator.Nelements)
    calculator.mass = zeros(calculator.Nelements)
    calculator.a = zeros(calculator.Nelements)
    calculator.lattice = ""
    d += 5

    # reads in the part of the eam file for each element
    for elem in 1:calculator.Nelements
        calculator.Z[elem] = parse(Int, data[d]) # the atomic number
        calculator.mass[elem] = parse(Float64, data[d+1]) # the atomic mass
        calculator.a[elem] = parse(Float64, data[d+2]) # the lattice constant
        calculator.lattice *= data[d+3] # the lattice type
        d += 4

        calculator.embedded_data[elem, :] = parse.(Float64, data[d:(d+calculator.nrho-1)]) # the embedded energy of the element
        d += calculator.nrho
        calculator.density_data[elem, :] = parse.(Float64, data[d:(d+calculator.nr-1)]) # the electron density of the element
        d += calculator.nr
    end

    # reads in the r*phi data for each interaction between elements
    calculator.rphi_data = zeros(calculator.Nelements, calculator.Nelements, calculator.nr)

    for i in 1:calculator.Nelements
        for j in 1:i
            calculator.rphi_data[j, i, :] = parse.(Float64, data[d:(d+calculator.nr-1)])
            d += calculator.nr
        end
    end

    calculator.r_range = (0:calculator.nr-1)*calculator.dr
    calculator.rho_range = (0:calculator.nrho-1)*calculator.drho
    calculator.r = collect(calculator.r_range)
    calculator.rho = collect(calculator.rho_range)

    set_splines(calculator)
end


# +
lines = readdlm("Al99.eam.alloy", '\n', String) # read the files, split by new line
calculator = EAM()

function lines_to_list(lines) # convert the entries in lines to list
    data = []
    for line in lines
        append!(data, split(line))
    end
    return data
end

i = 4 # start from the 4th line
data = lines_to_list(lines[i:end])

calculator.Nelements = parse(Int, data[1]) # number of elements
d = 2
calculator.elements = data[d:d+calculator.Nelements-1] # the elements symbols starts from the 2nd entries
d += calculator.Nelements

calculator.nrho = parse(Int, data[d]) 
calculator.drho = parse(Float64, data[d+1])
calculator.nr = parse(Int, data[d+2])
calculator.dr = parse(Float64, data[d+3])# the cutoff radius in angstroms
calculator.cutoff = parse(Float64, data[d+4]) 

calculator.embedded_data = zeros(calculator.Nelements, calculator.nrho)
calculator.density_data = zeros(calculator.Nelements, calculator.nr)
calculator.Z = zeros(Int, calculator.Nelements)
calculator.mass = zeros(calculator.Nelements)
calculator.a = zeros(calculator.Nelements)
calculator.lattice = ""
d += 5

# reads in the part of the eam file for each element
for elem in 1:calculator.Nelements
    calculator.Z[elem] = parse(Int, data[d]) # the atomic number
    calculator.mass[elem] = parse(Float64, data[d+1]) # the atomic mass
    calculator.a[elem] = parse(Float64, data[d+2]) # the lattice constant
    calculator.lattice *= data[d+3] # the lattice type
    d += 4

    calculator.embedded_data[elem, :] = parse.(Float64, data[d:(d+calculator.nrho-1)]) # the embedded energy of the element
    d += calculator.nrho
    calculator.density_data[elem, :] = parse.(Float64, data[d:(d+calculator.nr-1)]) # the electron density of the element
    d += calculator.nr
end

# reads in the r*phi data for each interaction between elements
calculator.rphi_data = zeros(calculator.Nelements, calculator.Nelements, calculator.nr)

for i in 1:calculator.Nelements
    for j in 1:i
        calculator.rphi_data[j, i, :] = parse.(Float64, data[d:(d+calculator.nr-1)])
        d += calculator.nr
    end
end

calculator.r_range = (0:calculator.nr-1)*calculator.dr
calculator.rho_range = (0:calculator.nrho-1)*calculator.drho
calculator.r = collect(calculator.r_range)
calculator.rho = collect(calculator.rho_range)

set_splines(calculator)

# -

# ## 1. Read potential

eam = EAM()
fname = "Al99.eam.alloy"
read_potential!(eam, fname)

# ## 2. Calculate potential energy

# +
function get_neighbors(neig, i)
    neighbors::Vector{Int} = []
    for j in 1:length(neig)
        neig_j = neig[j]
        if neig_j[1] == i
            append!(neighbors, neig_j[2])
        end
        if neig_j[2] == i
            append!(neighbors, neig_j[1])
        end
    end
    return unique(neighbors)
end

function get_neighbors_all(sys::Molly.System)
    neighbors_all = [Int[] for _ in 1:length(sys.atoms)]
    n_threads = 1
    neig = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    for i in 1:length(neig)
        pair_i = neig[i]
        append!(neighbors_all[pair_i[1]], pair_i[2])
        append!(neighbors_all[pair_i[2]], pair_i[1])
    end
    return neighbors_all
end

function get_type(index_list::Vector{Int64}, typelist::Vector{Int64})
    list_type_index = Vector{Int}(undef, length(index_list))
    for i in 1:length(index_list)
        list_type_index[i] = indexin(1, typelist)[1]
    end
    return list_type_index
end

"""
calculate_energy(eam::EAM, sys::Molly.System, neighbors_all)

Calculate the total energy of a system using the Embedded Atom Method (EAM).

# Arguments
- `eam::EAM`: The EAM calculator.
- `sys::Molly.System`: The system object containing atom coordinates and types.
- `neighbors_all`: A precomputed list of neighbors for each atom.

# Returns
- `energy::Float64`: The total energy of the system in electron volts (eV).
"""
function calculate_energy(eam::EAM, sys::Molly.System, neighbors_all)
    n_threads = 1
    typelist = [1]

    pair_energy::Float64 = 0.0
    embedding_energy::Float64 = 0.0
    total_density::Vector{Float64} = zeros(length(sys.atoms))

    # neig = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    # neighbors_all = get_neighbors_all(sys)
    
    i_type::Int = 1 in typelist ? indexin(1, typelist)[1] : error("1 not found in typelist")

    for i::Int in 1:length(sys.atoms)
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]
        coords_i = sys.coords[i]
        
        if isempty(neighbors)
            continue
        end

        n_neighbors::Int = length(neighbors)
        d_i::Vector{Float64} = zeros(n_neighbors)
        for (index_j, j::Int) in enumerate(neighbors)
            d_i[index_j] = ustrip(sqrt(sum(vector(coords_i, sys.coords[j], sys.boundary).^2)))*10
        end

        eam_phi_ix = eam.phi[i_type, :]
        eam_embedded_energy_i = eam.embedded_energy[i_type]
        for j_type::Int in 1:eam.Nelements
            # use = get_type(neighbors, typelist) .== j_type
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            d_use = d_i

            pair_energy += sum(eam_phi_ix[j_type].(d_use))

            # density = Float64(sum(eam.electron_density[j_type].(d_use)))
            total_density[i] += sum(eam.electron_density[j_type].(d_use))
        end
        embedding_energy += eam_embedded_energy_i.(total_density[i])
    end

    components = Dict("pair" => pair_energy/2, "embedding" => embedding_energy)
    energy::Float64 = sum(values(components))
    return energy*u"eV"
end

# +
atoms_ase_sim = convert_ase_custom(molly_system)

neighbors_all = get_neighbors_all(molly_system)

# run first time before timing
AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
calculate_energy(eam, molly_system, neighbors_all)

println("Calculating potential energy using ASE EAM calculator")
@time E_ASE = AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
println("Calculating potential energy using my EAM calculator")
@time E_my = calculate_energy(eam, molly_system, neighbors_all)
@printf("ASE EAM calculator: %e eV\n",ustrip(E_ASE))
@printf("My EAM calculator: %e eV\n",ustrip(E_my))
@printf("Difference: %e eV\n",ustrip(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal) - calculate_energy(eam, molly_system, neighbors_all)))

function eam_ASE()
    AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
end
function eam_my()
    calculate_energy(eam, molly_system, neighbors_all)
end

n_repeat = 10
t0 = time()
E_ASE = repeat(eam_ASE, n_repeat)
t1 = time()
E_ASE = repeat(eam_my, n_repeat)
t2 = time()

println("time/atom/step by ASE EAM calculator: ", (t1-t0)/n_repeat/length(molly_system.atoms), " seconds")
println("time/atom/step by my EAM calculator: ", (t2-t1)/n_repeat/length(molly_system.atoms), " seconds")
# -

using ProfileView
ProfileView.@profview repeat(eam_my, n_repeat)

# ## 3. Calculate force

"""
    calculate_forces(eam::EAM, sys::Molly.System, neighbors_all)

Calculate the forces on particles in a molecular system using the Embedded Atom Method (EAM).

# Arguments
- `eam::EAM`: An instance of the EAM potential.
- `sys::Molly.System`: The molecular system.
- `neighbors_all`: A matrix containing the neighbors of each particle in the system.

# Returns
- `forces_particle`: A matrix containing the forces on each particle in the system.
"""
function calculate_forces(eam::EAM, sys::Molly.System, neighbors_all)
    n_threads::Int = 1
    typelist::Vector{Int} = [1]

    i_type::Int = 1 in typelist ? indexin(1, typelist)[1] : error("1 not found in typelist")
    d_electron_density_i = eam.d_electron_density[i_type]
    
    forces_particle::Matrix{Float64} = zeros(length(sys.coords),3)
    
    # calculate total_density
    total_density::Vector{Float64} = zeros(length(sys.atoms))
    r_all::Vector{Matrix{Float64}} = []
    d_all::Vector{Vector{Float64}} = []

    n_neighbors_all::Vector{Int} = [length(neighbors_all[i]) for i in 1:length(sys.atoms)]
    n_neighbors_max::Int = maximum(n_neighbors_all)

    r_i::Matrix{Float64} = zeros(n_neighbors_max,3)
    d_i::Vector{Float64} = zeros(n_neighbors_max)
    for i::Int in 1:length(sys.atoms)
        # i_type::Int = indexin(1, typelist)[1]
        
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]    
        if isempty(neighbors)
            continue
        end

        n_neighbors::Int = length(neighbors)
        coords_i = sys.coords[i]
    
        # distance between atom i and its neighbors
        # r_i::Matrix{Float64} = zeros(n_neighbors,3)
        # d_i::Vector{Float64} = zeros(n_neighbors)
        for (index_j::Int, j::Int) in enumerate(neighbors)
            r_ij = ustrip(vector(coords_i, sys.coords[j], sys.boundary)*10) # convert to Å
            d_ij = sqrt(sum(r_ij.^2))
            r_i[index_j,1:3] = r_ij
            d_i[index_j] = d_ij
        end

        push!(r_all, r_i[1:n_neighbors,1:3])
        push!(d_all, d_i[1:n_neighbors])
    
        for j_type::Int in 1:eam.Nelements # iterate over all types
            # use = get_type(neighbors, typelist) .== j_type # get the index of the neighbors with type j
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            d_use = d_all[i]
    
            density = sum(eam.electron_density[j_type].(d_use)) # electron density
            total_density[i] += density # total electron density around atom i
        end
    end
    
    # calculate forces on particles
    for i::Int in 1:length(sys.coords)
        # i_type::Int = indexin(1, typelist)[1]
            
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]
        if isempty(neighbors)
            continue
        end
        n_neighbors::Int = length(neighbors)
        coords_i = sys.coords[i]
    
        r_i = r_all[i]
        d_i = d_all[i]
        
        # derivative of the embedded energy of atom i
        d_embedded_energy_i::Float64 = eam.d_embedded_energy[i_type].(total_density[i])
    
        ur_i = r_i
    
        # unit directional vector
        ur_i ./= d_i
        
        for j_type::Int in 1:eam.Nelements
            # use = get_type(neighbors, typelist) .== j_type # get the index of the neighbors with type j
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            # ur_use = ur_i[use, :]
            # neighbors_use = neighbors[use]
            d_use = d_i
            ur_use::Matrix{Float64} = ur_i[:,:]
            neighbors_use = neighbors
    
            total_density_j = total_density[neighbors_use]
            
            scale::Vector{Float64} = eam.d_phi[i_type, j_type].(d_use)
            scale .+= (d_embedded_energy_i .* eam.d_electron_density[j_type].(d_use)) 
            scale .+= (eam.d_embedded_energy[j_type].(total_density_j) .* d_electron_density_i.(d_use))
    
            forces_particle[i, :] .+= (ur_use' * scale)
        end
    end

    return [SVector{3,Float64}(forces_particle[idx_f,:]) for idx_f in 1:size(forces_particle)[1]]*u"eV/Å"
end

# +
# run first time before timing
forces_ASE = AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
forces_my = calculate_forces(eam, molly_system, neighbors_all)

println("Calculating forces using ASE EAM calculator")
@time forces_ASE = AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
println("Calculating forces using My EAM calculator")
@time forces_my = calculate_forces(eam, molly_system, neighbors_all)

@printf("Sum of forces by ASE EAM calculator: [%e %e %e] eV/Å\n",ustrip(sum(forces_ASE))...)
@printf("Sum of forces by my EAM calculator: [%e %e %e] eV/Å\n",ustrip(sum(forces_my))...)

forces_err = forces_my - forces_ASE
index_max_forces_err = argmax([sqrt(sum(fe.^2)) for fe in forces_err])
@printf("Max force error: %e eV/Å\n", ustrip(sqrt(sum(forces_err[index_max_forces_err].^2))))

function eam_ASE_f()
    AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
end
function eam_my_f()
    calculate_forces(eam, molly_system, neighbors_all)
end

eam_ASE_f()
eam_my_f()
n_repeat = 10
t0 = time()
E_ASE = repeat(eam_ASE_f, n_repeat)
t1 = time()
E_ASE = repeat(eam_my_f, n_repeat)
t2 = time()

println("time/atom/step by ASE EAM calculator: ", (t1-t0)/n_repeat/length(molly_system.atoms), " seconds")
println("time/atom/step by my EAM calculator: ", (t2-t1)/n_repeat/length(molly_system.atoms), " seconds")

# -

ProfileView.@profview repeat(eam_my_f, n_repeat)
@code_warntype repeat(eam_my_f, n_repeat)
