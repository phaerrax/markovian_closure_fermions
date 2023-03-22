using ITensors
using IterTools
using DelimitedFiles
using PseudomodesTTEDOPA

include("./TDVP_lib_VecRho.jl")

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    rcoups = thermofield_coefficients[:, 1]
    rfreqs = thermofield_coefficients[:, 3]
    lcoups = thermofield_coefficients[:, 2]
    lfreqs = thermofield_coefficients[:, 4]

    chain_length = parameters["chain_length"]
    systempos = chain_length + 1
    total_size = 2 * chain_length + 1
    leftchain_sites = (systempos - 1):-1:1
    rightchain_sites = (systempos + 1):1:total_size

    sites = siteinds("Fermion", total_size)
    psi0 = MPS(
        sites,
        [repeat(["Occ"], chain_length); system_initstate; repeat(["Emp"], chain_length)],
    )

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "n", systempos

    # - system-chain interaction
    if lowercase(parameters["interaction_type"]) == "xx"
        h += im*lcoups[1], "c", systempos, "c", leftchain_sites[1]
        h += im*lcoups[1], "c", systempos, "c†", leftchain_sites[1]
        h += im*lcoups[1], "c†", systempos, "c", leftchain_sites[1]
        h += im*lcoups[1], "c†", systempos, "c†", leftchain_sites[1]

        h += im*rcoups[1], "c", systempos, "c", rightchain_sites[1]
        h += im*rcoups[1], "c", systempos, "c†", rightchain_sites[1]
        h += im*rcoups[1], "c†", systempos, "c", rightchain_sites[1]
        h += im*rcoups[1], "c†", systempos, "c†", rightchain_sites[1]
    elseif lowercase(parameters["interaction_type"]) == "exchange"
        h += lcoups[1], "c†", systempos, "c", leftchain_sites[1]
        h += lcoups[1], "c†", leftchain_sites[1], "c", systempos

        h += rcoups[1], "c†", systempos, "c", rightchain_sites[1]
        h += rcoups[1], "c†", rightchain_sites[1], "c", systempos
    else
        throw(error("Unrecognized interaction type. Please use \"xx\" or \"exchange\"."))
    end

    # - chain terms
    for (j, site) in enumerate(leftchain_sites)
        h += lfreqs[j], "n", site
    end
    for (j, (site1, site2)) in enumerate(partition(leftchain_sites, 2, 1))
        h += lcoups[j + 1], "c†", site1, "c", site2
        h += lcoups[j + 1], "c†", site2, "c", site1
    end

    for (j, site) in enumerate(rightchain_sites)
        h += rfreqs[j], "n", site
    end
    for (j, (site1, site2)) in enumerate(partition(rightchain_sites, 2, 1))
        h += rcoups[j + 1], "c†", site1, "c", site2
        h += rcoups[j + 1], "c†", site2, "c", site1
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
