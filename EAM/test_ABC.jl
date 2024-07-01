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

# -

# ## Define interaction

# +
## 1. Import ASE and other Python modules
# Import ASE and other Python modules as needed
ase = pyimport("ase")
ase_view = pyimport("ase.visualize")
ase_plot = pyimport("ase.visualize.plot")
plt = pyimport("matplotlib.pyplot")

# Import Al EAM potential
fname = "Al99.eam.alloy"
EAM = pyimport("ase.calculators.eam") # python ASE-EAM calculator
eam_cal = ASEconvert.ASEcalculator(EAM.EAM(potential=fname))  # EAM calculater, converted to AtomsBase format

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

# ## Create Molly system

# +
## 3. Build an aluminum surface with adsorbate
al_LatConst = 4.0495
atom_mass = 26.9815u"u"  # Atomic mass of aluminum in grams per mole

function system_adatom(size)

    # Build an (001) Al surface  
    atoms_ase = ase.build.fcc100("Al", size=size, vacuum = al_LatConst*2)
    # The basis vectors on x and y are along 1/2<110> directions
    ase.build.add_adsorbate(atoms_ase, "Al", al_LatConst/2, position=(al_LatConst*(2.5*sqrt(1/2)/2),al_LatConst*(2.5*sqrt(1/2)/2)))
    # ase.build.add_adsorbate(atoms_ase, "Al", al_LatConst/2, "bridge")

    atoms_ase.translate([al_LatConst*(sqrt(1/2)/4),al_LatConst*(sqrt(1/2)/4),0])
    atoms_ase.wrap()

    atoms_ase_cell = atoms_ase.get_cell()
    box_size = pyconvert(Array{Float64}, [atoms_ase_cell[x,x] for x in range(0,2)])*0.1*u"nm"

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
# -

# ## Initialize simulation

# ### Initialize System

# +
molly_atoms, atoms_ab, box_size, atom_positions = system_adatom((4,4,6))

# interactions
# EAMInteraction with the ASE EAM calculator and system representation
eam_interaction = EAMInteraction(eam_cal, atoms_ab)

# Specify boundary condition
boundary_condition = Molly.CubicBoundary(box_size[1],box_size[2],box_size[3])

# Create the Molly System with atoms, positions, velocities, and boundary
molly_system = Molly.System(
    atoms=molly_atoms,
    atoms_data = [AtomData(element="Al") for a in molly_atoms],
    coords=atom_positions,  # Ensure these are SVector with correct units
    boundary=boundary_condition,
    general_inters=[eam_interaction],  # This needs to be filled with actual interaction objects compatible with Molly
    # loggers=Dict(:kinetic_eng => Molly.KineticEnergyLogger(100), :pot_eng => Molly.PotentialEnergyLogger(100)),
    loggers=(coords=CoordinateLogger(1),),
    energy_units=u"eV",  # Ensure these units are correctly specified
    force_units=u"eV/nm"  # Ensure these units are correctly specified
    )

atom_positions_init = copy(atom_positions)
function initialize_system(loggers=(coords=CoordinateLogger(1),))
    # Initialize the system with the initial positions and velocities
    system_init = Molly.System(
    atoms=molly_atoms,
    atoms_data = [AtomData(element="Al") for a in molly_atoms],
    coords=atom_positions_init,  # Ensure these are SVector with correct units
    boundary=boundary_condition,
    general_inters=[eam_interaction],  # This needs to be filled with actual interaction objects compatible with Molly
    # loggers=Dict(:kinetic_eng => Molly.KineticEnergyLogger(100), :pot_eng => Molly.PotentialEnergyLogger(100)),
    loggers=loggers,
    energy_units=u"eV",  # Ensure these units are correctly specified
    force_units=u"eV/nm"  # Ensure these units are correctly specified
    )
    return system_init
end
# -

# ### Define ABCSimulator

# +
# Define the ABCSimulator structure
"""
In the constructor function ABCSimulator, default values are provided for each of these fields. 
If you create a SteepestDescentMinimizer without specifying the types, default values 
will determine the types of the fields. For example, if you create a ABCSimulator without specifying sigma, 
it will default to 0.01*u"nm", and S will be the type of this value.
"""
struct ABCSimulator{S,W,D,F,L}
    sigma::S 
    W::W
    max_steps::Int
    max_steps_minimize::Int
    step_size_minimize::D
    tol::F
    log_stream::L
end

"""
ABCSimulator(; sigma=0.01*u"nm", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.01u"nm", tol=1e-10u"kg*m*s^-2", log_stream=devnull)

Constructor for ABCSimulator.

## Arguments
- `sigma`: The value of sigma in units of nm.
- `W`: The value of W in units of eV.
- `max_steps`: The maximum number of steps for the simulator.
- `max_steps_minimize`: The maximum number of steps for the minimizer.
- `step_size_minimize`: The step size for the minimizer in units of nm.
- `tol`: The tolerance for convergence in units of kg*m*s^-2.
- `log_stream`: The stream to log the output.

## Returns
- An instance of ABCSimulator.
"""
function ABCSimulator(;
                        sigma=0.01*u"nm", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.01u"nm",tol=1e-10u"kg*m*s^-2",
                        log_stream=devnull)
    return ABCSimulator(sigma, W, max_steps, max_steps_minimize, step_size_minimize, tol, log_stream)
end

# Penalty function with Gaussuan form
"""
Returns a penalty function of system coordinate x with Gaussuan form
x:      System coordinate
x_0:    Reference system coordinate
sigma:  Spatial extent of the activation, per sqrt(degree of freedom)
W:      Strenth of activation, per degree of freedom
pbc:    Periodic boundary conditions
"""
function f_phi_p(x, x_0, sigma, W, pbc=nothing)
    N = length(x)
    
    sigma2_new = sigma^2*3*N
    EDSQ = (A, B) -> sum(sum(map(x -> x.^2, A-B)))
    phi_p = sum([W * exp(-EDSQ(x,c) / (2*sigma2_new)) for c in x_0])
    return phi_p
end

# Calculate the gradient of the penalty energy
function penalty_forces(sys, penalty_coords, sigma, W)
    # Function of the penalty energy for a given coordinate
    f_phi_p_coords = x -> f_phi_p(x, penalty_coords, sigma, W)

    # Calculate the gradient of the penalty energy, The penalty force is the negative gradient of the penalty energy
    penalty_fs = -gradient(f_phi_p_coords, sys.coords)[1]

    return penalty_fs
end

# Define the forces function with penalty term
"""
Evaluate the forces acting on the system with penalty term
If there is no penalty term, the penalty_coords should be set to nothing, 
and return the forces identical to the original forces function
"""
function Molly.forces(sys::System, penalty_coords, sigma, W, neighbors;
    n_threads::Integer=Threads.nthreads()) 

    
    fs = forces(sys, neighbors; n_threads=n_threads)

    # Add penalty term to forces
    if penalty_coords != nothing
        fs += penalty_forces(sys, penalty_coords, sigma, W)
    end

    return fs
end

# Define the Minimize! function, on the basis of the Molly SteepestDescentMinimizer
"""
Minimize!(sys, sim, penalty_coords; n_threads::Integer, frozen_atoms=[])

Minimizes the system `sys` energy using the simulatior `sim` providing penalty coordinates `penalty_coords`.

# Arguments
- `sys`: The system to be minimized.
- `sim`: The simulation object.
- `penalty_coords`: The penalty coordinates used in the minimization.
- `n_threads`: The number of threads to use in the minimization.
- `frozen_atoms`: (optional) A list of atoms to be frozen during the minimization.

# Examples
"""
function Minimize!(sys, sim, penalty_coords; n_threads::Integer, frozen_atoms=[])
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    using_constraints = length(sys.constraints) > 0
    hn = sim.step_size_minimize
    E_phi = 0*u"eV"
    if penalty_coords!=nothing
        E_phi += f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W)
    end
    E = potential_energy(sys, neighbors; n_threads=n_threads) + E_phi
    
    # Set F_multiplier of frozen_atoms to zero
    F_multiplier = ones(length(sys.coords))

    for atom in frozen_atoms
        F_multiplier[atom] = 0
    end

    for step_n in 1:sim.max_steps_minimize
        # Calculate the forces using the new forces function
        # penalty_coords is fixed throughout the minimization
        F = forces(sys, penalty_coords, sim.sigma, sim.W, neighbors; n_threads=1)
        F = F.*F_multiplier
        max_force = maximum(norm.(F))
        
        coords_copy = sys.coords
        sys.coords += hn * F ./ max_force
        using_constraints && apply_position_constraints!(sys, coords_copy; n_threads=n_threads)

        neighbors_copy = neighbors
        
        # System energy after applying displacements in this step
        E_phi = 0*u"eV"
        if penalty_coords!=nothing
            E_phi += f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W)
        end

        E_trial = potential_energy(sys, neighbors; n_threads=n_threads) + E_phi
        if E_trial < E
            hn = 6 * hn / 5
            E = E_trial
            # println(sim.log_stream, "Step ", step_n, " - potential energy ",
            #         E_trial, " - max force ", max_force, " - accepted")
        else
            sys.coords = coords_copy
            neighbors = neighbors_copy
            hn = hn / 5
            # println(sim.log_stream, "Step ", step_n, " - potential energy ",
            #         E_trial, " - max force ", max_force, " - rejected")
        end

        if max_force < sim.tol
            break
        end
    end
    F = forces(sys, penalty_coords, sim.sigma, sim.W, neighbors; n_threads=1)
    F = F.*F_multiplier
    max_force = maximum(norm.(F))
    @printf("max force = %e N ",ustrip(max_force))
    return sys
end

# Implement the simulate! function for ABCSimulator
"""
simulate!(sys, sim::ABCSimulator; n_threads::Integer=Threads.nthreads(), frozen_atoms=[], run_loggers=true, fname="output.txt")

Simulates the system using the ABCSimulator.

# Arguments
- `sys`: The system to be simulated.
- `sim`: An instance of the ABCSimulator.
- `n_threads`: The number of threads to use for parallel execution. Defaults to the number of available threads.
- `frozen_atoms`: A list of atoms that should be frozen during the simulation.
- `run_loggers`: A boolean indicating whether to run the loggers during the simulation.
- `fname`: The name of the output file.

# Examples 
simulate!(molly_system, simulator, n_threads=1, fname="output_test.txt", frozen_atoms=frozen_atoms)
"""
function simulate!(sys, sim::ABCSimulator; n_threads::Integer=Threads.nthreads(), frozen_atoms=[], run_loggers=true, fname="output.txt")

    # Do not wrap the coordinates
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)

    # open an empty output file
    open(fname, "w") do file
        write(file, "")
    end

    # Set d_multiplier of frozen_atoms to zero
    d_multiplier = ones(length(sys.coords))
    for atom in frozen_atoms
        d_multiplier[atom] = 0
    end

    # 0. Call Minimize! without penalty_coords before the loop
    Minimize!(sys, sim, nothing; n_threads=n_threads, frozen_atoms=frozen_atoms)
    E = potential_energy(sys, neighbors; n_threads=n_threads)

    # Run the loggers, Log the step number (or other details as needed)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    @printf("step %d: ",0)
    print(E)
    print("\n")

    # 1. Store the initial coordinates
    penalty_coords = [copy(sys.coords)]  

    for step_n in 1:sim.max_steps
        # 2. Slightly perturb the system coordinates
        for i in 1:length(sys.coords)
            random_direction = randn(size(sys.coords[i]))
            sys.coords[i] += 1e-4*u"nm" * random_direction*d_multiplier[i]
        end

        # 3. Call Minimize! with penalty_coords, update system coordinates
        Minimize!(sys, sim, penalty_coords; n_threads=n_threads, frozen_atoms=frozen_atoms)
        E = potential_energy(sys, neighbors; n_threads=n_threads)
        @printf("step %d: ",step_n)
        print(E)
        print("\n")

        # Run the loggers, Log the step number (or other details as needed)
        run_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads)
        println(sim.log_stream, "Step 0 - potential energy ",
                E, " - max force N/A - N/A")
        E_phi = f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W)
        open(fname, "a") do file
            write(file, string(ustrip(E))*" "*string(ustrip(E_phi))*"\n")
        end

        # 4. Update penalty_coords for the next step
        push!(penalty_coords, copy(sys.coords))
    end
    return sys
end
# -

# ## Update System with ABCSimulator

# +
molly_system = initialize_system()

# Start from an energy minimum
# tol: the default value was 1000 kJ/mol/nm ~= 9.6e13 eV/m ~= 1.5e-5 J/M
simulator = SteepestDescentMinimizer(step_size=1e-3u"nm", tol=1e-12u"kg*m*s^-2", log_stream=devnull)
Molly.simulate!(molly_system, simulator)
atoms_ase_sim = convert_ase_custom(molly_system)
print(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal))
print("\n")
# -

# ### Specify the atoms to be frozen

z_coords = [coords[3] for coords in molly_system.coords]
frozen_atoms = [index for (index, z_coord) in enumerate(z_coords) if z_coord < al_LatConst*2.9*0.1*u"nm"]
print(length(frozen_atoms))
print("\n")

# ### Run ABCSimulator

# +
sigma = 2e-3
W = 0.1
@printf("sigma = %e nm/dof^1/2\n W = %e eV",ustrip(sigma),ustrip(W))

simulator = ABCSimulator(sigma=sigma*u"nm", W=W*u"eV", max_steps=1, max_steps_minimize=60, step_size_minimize=1.5e-3u"nm", tol=1e-12u"kg*m*s^-2")
# Run the simulation
print("\n")
simulate!(molly_system, simulator, n_threads=1, fname="output_test.txt", frozen_atoms=frozen_atoms)

# # simulation cell after energy minimization
atoms_ase_sim = convert_ase_custom(molly_system)
ase_view.view(atoms_ase_sim, viewer="x3d")
# -

## visualize
using GLMakie
color_0 = :blue
color_1 = :red
colors = []
for (index, value) in enumerate(molly_system.coords)
    push!(colors, index < length(molly_system.coords) ? color_0 : color_1)
end
visualize(molly_system.loggers.coords, boundary_condition, "test.mp4", markersize=0.1, color=colors)
