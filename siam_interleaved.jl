using ITensors
using IterTools
using DelimitedFiles
using PseudomodesTTEDOPA

include("./TDVP_lib_VecRho.jl")

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.
# The chains are then interleaved, so that we end up with one chain only (here we are
# still dealing with the spinless case).
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
    #filledchain_sites = (systempos - 1):-1:1
    #emptychain_sites = (systempos + 1):1:total_size
    filledchain_sites = 3:2:total_size
    emptychain_sites = 2:2:total_size

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            systempos => parameters["sys_ini"]
            [st => "Occ" for st in filledchain_sites]
            [st => "Emp" for st in emptychain_sites]
        ],
    )
    psi0 = MPS(sites, [initialsites[i] for i in 1:total_size])

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "n", systempos

    # - system-chain interaction
    h += filledcoups[1], "c†", systempos, "c", filledchain_sites[1]
    h += filledcoups[1], "c†", filledchain_sites[1], "c", systempos

    h += emptycoups[1], "c†", systempos, "c", emptychain_sites[1]
    h += emptycoups[1], "c†", emptychain_sites[1], "c", systempos

    # - chain terms
    for (j, site) in enumerate(filledchain_sites)
        h += filledfreqs[j], "n", site
    end
    for (j, (site1, site2)) in enumerate(partition(filledchain_sites, 2, 1))
        h += filledcoups[j + 1], "c†", site1, "c", site2
        h += filledcoups[j + 1], "c†", site2, "c", site1
    end

    for (j, site) in enumerate(emptychain_sites)
        h += emptyfreqs[j], "n", site
    end
    for (j, (site1, site2)) in enumerate(partition(emptychain_sites, 2, 1))
        h += emptycoups[j + 1], "c†", site1, "c", site2
        h += emptycoups[j + 1], "c†", site2, "c", site1
    end

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

    tdvp1!(
        psi,
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
end
