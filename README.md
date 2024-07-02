# ABCD_J: Autonomous Basin Climbing Dynamics in Julia

Incorporating Julia into a new metadynamics molecular simulation program. The simulation framework was developed using the Julia packages [Molly.jl](https://github.com/JuliaMolSim/Molly.jl.git), [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl.git), and [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl).

## `EAM/`: Incoporate EAM forcefield to benchmark the Al adatom toy model.

### `test_minimize.ipynb`: Incorporated the EAM forcefield for metallic element interactions from Python [ASE](https://gitlab.com/ase/ase.git) package to the Molly system/simulator framework.

![](https://github.com/ch-tung/ABCD_J/blob/d94d2efdf2ac7988aea6161c3682c5a385521688/EAM.png?raw=true)

---

### `test_ABC.ipynb`: Defining custom ABC simulator function.

![](https://github.com/ch-tung/ABCD_J/blob/7fe5081cf97966e08ad64c2363283cd5736bd206/ABC.png?raw=true)

- Constructor for ABCSimulator.

`ABCSimulator(; sigma=0.01*u"nm", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.01u"nm",tol=1e-10u"kg*m*s^-2",`

`log_stream=devnull)`

Arguments

`sigma`: The value of sigma in units of nm.

`W`: The value of W in units of eV.

`max_steps`: The maximum number of steps for the simulator.

`max_steps_minimize`: The maximum number of steps for the minimizer.

`step_size_minimize`: The step size for the minimizer in units of nm.

`tol`: The tolerance for convergence in units of kg\*m\*s^-2.

`log_stream`: The stream to log the output.

- Simulates the system using the `ABCSimulator`.

`simulate!(sys, sim::ABCSimulator; n_threads::Integer=Threads.nthreads(), frozen_atoms=[], run_loggers=true, fname="output.txt")`

Arguments

`sys`: The system to be simulated.

`sim`: An instance of the ABCSimulator.

`n_threads`: The number of threads to use for parallel execution. Defaults to the number of available threads.

`frozen_atoms`: A list of atoms that should be frozen during the simulation.

`run_loggers`: A boolean indicating whether to run the loggers during the simulation.

`fname`: The name of the output file.

Example

```
molly_system = initialize_system()

# 1. Start from an energy minimum
simulator = SteepestDescentMinimizer(step_size=1e-3u"nm", tol=1e-12u"kg*m*s^-2", log_stream=devnull)
Molly.simulate!(molly_system, simulator)
atoms_ase_sim = convert_ase_custom(molly_system)
print(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal))
print("\n")

# 2. Specify the atoms to be frozen
z_coords = [coords[3] for coords in molly_system.coords]
frozen_atoms = [index for (index, z_coord) in enumerate(z_coords) if z_coord < al_LatConst*2.9*0.1*u"nm"]
print(length(frozen_atoms))
print("\n")

# 3. Run ABCSimulator
sigma = 2e-3
W = 0.1
@printf("sigma = %e nm/dof^1/2\n W = %e eV",ustrip(sigma),ustrip(W))
simulator = ABCSimulator(sigma=sigma*u"nm", W=W*u"eV", max_steps=100, max_steps_minimize=60, step_size_minimize=1.5e-3u"nm", tol=1e-12u"kg*m*s^-2")
print("\n")
simulate!(molly_system, simulator, n_threads=1, fname="output_test.txt", frozen_atoms=frozen_atoms)

# 4. visualize
using GLMakie
color_0 = :blue
color_1 = :red
colors = []
for (index, value) in enumerate(molly_system.coords)
    push!(colors, index < length(molly_system.coords) ? color_0 : color_1)
end
visualize(molly_system.loggers.coords, boundary_condition, "test.mp4", markersize=0.1, color=colors)
```

---

`test_JuliaEAM.ipynb`: Calculate EAM interactions using Julia.

- Read the potential data from a file and populates the fields of the `calculator` object.

`read_potential!(calculator::EAM, fd::String)`

Arguments

`calculator`: The EAM calculator object to populate with potential data.

`fd`: The file path to the potential data file.

- Calculate the total energy of a system using the Embedded Atom Method (EAM)

`calculate_energy(eam::EAM, sys::Molly.System, neighbors_all)`

Arguments

`eam`: The EAM potential parameters.

`sys`: The system object containing atom coordinates and types.

`neighbors_all`: A precomputed list of neighbors for each atom.

Returns

`energy`: The total energy of the system in electron volts (eV).

- Calculate the forces on particles in a molecular system using the Embedded Atom Method (EAM).

`calculate_forces(eam::EAM, sys::Molly.System, neighbors_all)`

Arguments

`eam`: An instance of the EAM potential.

`sys`: The molecular system.

`neighbors_all`: A precomputed list of neighbors for each atom.

Returns

`forces_particle`: A matrix containing the forces on each particle in the system.

Example

```
# 1. Read potential
eam = EAM()
fname = "Al99.eam.alloy"
read_potential!(eam, fname)

# 2. Calculate potential energy
atoms_ase_sim = convert_ase_custom(molly_system)
neighbors_all = get_neighbors_all(sys)

println("Calculating potential energy using ASE EAM calculator")
@time E_ASE = AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)

println("Calculating forces using my EAM calculator")
@time E_my = calculate_energy(eam, molly_system, neighbors_all)

@printf("ASE EAM calculator: %e eV\n",ustrip(E_ASE))
@printf("My EAM calculator: %e eV\n",ustrip(E_my))
@printf("Difference: %e eV\n",ustrip(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal) - calculate_energy(eam, molly_system, neighbors_all)))

## 3. Calculate force
println("Calculating forces using ASE EAM calculator")
@time forces_ASE = AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)

println("Calculating forces using my EAM calculator")
@time forces_my = calculate_forces(eam, molly_system, neighbors_all)

@printf("Sum of forces by ASE EAM calculator: [%e %e %e] eV/Å\n",ustrip(sum(forces_ASE))...)
@printf("Sum of forces by My EAM calculator: [%e %e %e] eV/Å\n",ustrip(sum(forces_my))...)

forces_err = forces_my - forces_ASE
index_max_forces_err = argmax([sqrt(sum(fe.^2)) for fe in forces_err])
@printf("Max force error: %e eV/Å\n", ustrip(sqrt(sum(forces_err[index_max_forces_err].^2))))
```

Outputs

```
Calculating potential energy using ASE EAM calculator
  0.049684 seconds (1.43 M allocations: 66.706 MiB)
Calculating potential energy using my EAM calculator
  0.023109 seconds (162.72 k allocations: 17.463 MiB)
ASE EAM calculator: -3.187108e+04 eV
My EAM calculator: -3.187108e+04 eV
Difference: 7.275958e-12 eV

Calculating forces using ASE EAM calculator
  0.057532 seconds (1.43 M allocations: 67.147 MiB, 14.76% gc time)
Calculating forces using My EAM calculator
  0.609485 seconds (19.73 M allocations: 1.187 GiB, 15.94% gc time)
Sum of forces by ASE EAM calculator: [-2.368150e-12 -2.400567e-12 2.278473e-14] eV/Å
Sum of forces by my EAM calculator: [-3.226586e-16 6.302250e-15 -1.188806e-14] eV/Å
Max force error: 2.002796e-12 eV/Å
```
