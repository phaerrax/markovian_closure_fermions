using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]
    delta = parameters["sys_coup"]

    # Input: chain parameters
    rcoefficients = readdlm(
        parameters["tedopa_coefficients_upper"], ',', Float64; skipstart=1
    )
    rcoups = rcoefficients[:, 1]
    rfreqs = rcoefficients[:, 2]

    lcoefficients = readdlm(
        parameters["tedopa_coefficients_lower"], ',', Float64; skipstart=1
    )
    lcoups = lcoefficients[:, 1]
    lfreqs = lcoefficients[:, 2]

    chain_length = parameters["chain_length"]
    systempos = chain_length + 1
    total_size = 2 * chain_length + 1
    leftchain_sites = (systempos - 1):-1:1
    rightchain_sites = (systempos + 1):1:total_size

    sites = siteinds("S=1/2", total_size)
    psi0 = productMPS(
        sites,
        [repeat(["Up"], chain_length); system_initstate; repeat(["Dn"], chain_length)],
    )

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "N", systempos
    h += delta, "σx", systempos

    # - system-chain interaction
    if lowercase(parameters["interaction_type"]) == "xx"
        # The interaction with both chains is
        #   Vs Vc = Vs Vc1 + Vs Vc2.
        # With the Jordan-Wigner transformation,
        #   i(c0 + c0†)(c1 + c1†) = - σy ⊗ σx.
        # We need to be careful with the ordering of the sites: if the sites of the chain
        # "1" are put to the left of the system, then we start from
        #   Vs Vc1 = i(c0 + c0†)(c-1 + c-1†) =
        #          = -i(c-1 + c-1†)(c0 + c0†) =
        #          = σy ⊗ σx.
        h += 4 * lcoups[1], "Sy", leftchain_sites[1], "Sx", systempos
        h += -4 * rcoups[1], "Sy", systempos, "Sx", rightchain_sites[1]
    elseif lowercase(parameters["interaction_type"]) == "exchange"
        h += lcoups[1], "S+", systempos, "S-", leftchain_sites[1]
        h += lcoups[1], "S-", systempos, "S+", leftchain_sites[1]
        h += rcoups[1], "S+", systempos, "S-", rightchain_sites[1]
        h += rcoups[1], "S-", systempos, "S+", rightchain_sites[1]
    else
        throw(error("Unrecognized interaction type. Please use \"xx\" or \"exchange\"."))
    end

    # - chain terms
    for j in 1:chain_length
        h += lfreqs[j], "N", leftchain_sites[j]
        h += rfreqs[j], "N", rightchain_sites[j]
    end

    for j in 1:(chain_length - 1)
        h += lcoups[j + 1], "S-", leftchain_sites[j], "S+", leftchain_sites[j + 1]
        h += lcoups[j + 1], "S+", leftchain_sites[j], "S-", leftchain_sites[j + 1]
        h += rcoups[j + 1], "S+", rightchain_sites[j], "S-", rightchain_sites[j + 1]
        h += rcoups[j + 1], "S-", rightchain_sites[j], "S+", rightchain_sites[j + 1]
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
        normalize=true,
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
