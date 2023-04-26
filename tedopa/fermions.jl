using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA

include("./TDVP_lib_VecRho.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]
    delta = parameters["sys_coup"]

    # Input: TTEDOPA chain parameters
    tedopa_coefficients = readdlm(
        parameters["tedopa_coefficients"], ',', Float64; skipstart=1
    )
    coups = tedopa_coefficients[:, 1]
    freqs = tedopa_coefficients[:, 2]
    chain_length = parameters["chain_length"]

    sites = siteinds("S=1/2", system_length + chain_length)
    psi0 = productMPS(sites, [system_initstate; repeat(["Dn"], chain_length)])

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "N", 1
    h += delta, "σx", 1

    # - system-chain interaction
    if lowercase(parameters["interaction_type"]) == "xx"
        # The XX interaction with the chain, with the Jordan-Wigner transformation, is
        #   i(c0 + c0†)(c1 + c1†) = - σy ⊗ σx.
        h += -4 * coups[1], "Sy", 1, "Sx", 2
    elseif lowercase(parameters["interaction_type"]) == "exchange"
        h += coups[1], "S+", 1, "S-", 2
        h += coups[1], "S-", 1, "S+", 2
    else
        throw(error("Unrecognized interaction type. Please use \"xx\" or \"exchange\"."))
    end

    # - TTEDOPA chain
    for j in system_length .+ (1:chain_length)
        h += freqs[j - 1], "N", j
    end

    for j in system_length .+ (1:(chain_length - 1))
        h += coups[j], "S-", j, "S+", j + 1
        h += coups[j], "S+", j, "S-", j + 1
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
        store_psi0=true,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )
end
