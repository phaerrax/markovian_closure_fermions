using ITensors
using ITensors.HDF5
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a spin-1/2 fermionic thermal bath, which is mapped onto
# two discrete chains by means of a thermofield+TEDOPA transformation.
# The chains are then interleaved, so that we end up with one chain only.
# Each site represent a (↑, ↓) fermion pair (i.e. has physical dimension 4).
#
# This script employs a flexible numbering of the sites, so that it is only necessary
# to change which sites are associated to the two (initially filled or empty) chains
# through the `systempos`, `filledchain_sites`, `emptychain_sites` variable, and the rest
# follows automatically.

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_length = 1
    eps = parameters["sys_en"]
    U = parameters["spin_interaction"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]

    chain_length = parameters["chain_length"]
    total_size = 2 * chain_length + 1
    systempos = 1
    filledchain_sites = 3:2:total_size
    emptychain_sites = 2:2:total_size

    initstate_file = get(parameters, "initial_state_file", nothing)
    if isnothing(initstate_file)
        sites = siteinds("Electron", total_size)
        initialsites = Dict(
            [
                systempos => parameters["sys_ini"]
                [st => "UpDn" for st in filledchain_sites]
                [st => "Emp" for st in emptychain_sites]
            ],
        )
        ψ = MPS(sites, [initialsites[i] for i in 1:total_size])
        start_from_file = false
    else
        ψ = h5open(initstate_file, "r") do file
            return read(file, parameters["initial_state_label"], MPS)
        end
        sites = siteinds(ψ)
        start_from_file = true
        # We need to extract the site indices from ψ or else, if we define them from
        # scratch, they will have different IDs and they won't contract correctly.
    end

    # Hamiltonian of the system:
    h = OpSum()

    h += eps, "n↓", systempos
    h += eps, "n↑", systempos
    h += U, "n↑ * n↓", systempos

    h += emptycoups[1] * exchange_interaction(sites[systempos], sites[emptychain_sites[1]])
    h +=
        filledcoups[1] * exchange_interaction(sites[systempos], sites[filledchain_sites[1]])
    h += spin_chain(emptyfreqs, emptycoups, sites[emptychain_sites])
    h += spin_chain(filledfreqs, filledcoups, sites[filledchain_sites])

    H = MPO(h, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        if !start_from_file
            growMPS!(ψ, parameters["max_bond"])
        end
        tdvp1!(
            ψ,
            H,
            timestep,
            tmax;
            hermitian=true,
            normalize=false,
            callback=cb,
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            store_psi0=false,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    else
        @info "Using adaptive algorithm."
        if !start_from_file
            growMPS!(ψ, 4)
        end
        adaptivetdvp1!(
            ψ,
            H,
            timestep,
            tmax;
            hermitian=true,
            normalize=false,
            callback=cb,
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            store_psi0=false,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
        )
    end
end
