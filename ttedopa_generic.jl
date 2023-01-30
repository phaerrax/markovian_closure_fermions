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
    coups = readdlm(parameters["chain_freqs"]) # Coupling constants
    freqs = readdlm(parameters["chain_coups"]) # Frequencies
    chain_length = parameters["chain_length"]

    sites = siteinds("S=1/2", system_length + chain_length)
    psi0 = productMPS(sites, [system_initstate; repeat(["Dn"], chain_length)])

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, 20)

    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += 2eps, "N", 1

    # - system-chain interaction
    h += 4coups[1], "S+", 1, "S-", 2
    h += 4coups[1], "S-", 1, "S+", 2

    # - TTEDOPA chain
    for j in system_length .+ (1:chain_length)
        h += freqs[j - 1], "N", j
    end

    for j in system_length .+ (1:(chain_length - 1))
        h += 4coups[j], "S-", j, "S+", j + 1
        h += 4coups[j], "S+", j, "S-", j + 1
    end

    H = MPO(h, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = createObs([
        ["Norm", 1],
        ["N", 1],
        ["N", 10],
        ["N", 20],
    ])
    # Every site is a S=1/2, so we can use a LocalMeasurementCallback without worries.
    cb = LocalPosMeasurementCallback(obs, sites, parameters["ms_stride"] * timestep)

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