using ITensors, ITensorMPS
using DelimitedFiles
using LindbladVectorizedTensors
using TimeEvoVecMPS
using IterTools

# This script tries to emulate the simulation of the interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a spin-1/2 fermionic thermal bath, which is mapped onto
# two discrete chains by means of a thermofield+TEDOPA transformation.
# One chain represent the modes above the chemical potential, the other one the modes below
# it.
# On each site we can have up to two particles, with opposite spins: each site is a
# (vectorized) space of a (↑, ↓) fermion pair.

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    system_initstate = parameters["sys_ini"]
    system_length = 1
    ε = parameters["sys_en"]
    U = parameters["spin_interaction"]

    # Input: chain stub parameters
    # ----------------------------
    thermofield_coefficients = readdlm(
        parameters["chain_coefficients"], ',', Float64; skipstart=1
    )
    empty_chain_coups = thermofield_coefficients[:, 1]
    empty_chain_freqs = thermofield_coefficients[:, 3]
    filled_chain_coups = thermofield_coefficients[:, 2]
    filled_chain_freqs = thermofield_coefficients[:, 4]
    chain_length = parameters["chain_length"]

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    filled_chain_range = range(; start=3, step=2, length=chain_length)

    total_size = system_length + 2chain_length

    sites = siteinds("vElectron", total_size)
    initialstates = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "UpDn" for st in filled_chain_range]
            [st => "Emp" for st in empty_chain_range]
        ],
    )
    initialops = Dict(
        [
            system_site => "v" * parameters["system_observable"]
            [st => "vId" for st in filled_chain_range]
            [st => "vId" for st in empty_chain_range]
        ],
    )
    init_state = MPS(sites, [initialstates[i] for i in 1:total_size])
    init_targetop = MPS(sites, [initialops[i] for i in 1:total_size])

    # Unitary part of master equation
    # -------------------------------
    adjℓ = OpSum()

    adjℓ += -ε * gkslcommutator("Ntot", system_site)
    adjℓ += -U * gkslcommutator("NupNdn", system_site)

    adjℓ +=
        empty_chain_coups[1] *
        exchange_interaction′(sites[system_site], sites[empty_chain[1]])

    adjℓ +=
        filled_chain_coups[1] *
        exchange_interaction′(sites[system_site], sites[filled_chain[1]])

    adjℓ += spin_chain′(
        empty_chain_freqs[1:chain_length],
        empty_chain_coups[2:chain_length],
        sites[empty_chain_range],
    )
    adjℓ += spin_chain′(
        filled_chain_freqs[1:chain_length],
        filled_chain_coups[2:chain_length],
        sites[filled_chain_range],
    )

    adjL = MPO(adjℓ, sites)

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        targetop, _ = stretchBondDim(init_targetop, parameters["max_bond"])
        adjtdvp1vec!(
            targetop,
            init_state,
            adjL,
            parameters["tstep"],
            parameters["tmax"],
            parameters["ms_stride"] * parameters["tstep"],
            sites;
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    else
        @info "Using adaptive algorithm."
        targetop, _ = stretchBondDim(init_targetop, 4)
        adaptiveadjtdvp1vec!(
            targetop,
            init_state,
            adjL,
            parameters["tstep"],
            parameters["tmax"],
            parameters["ms_stride"] * parameters["tstep"],
            sites;
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
        )
    end
end
