using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
using IterTools

include("../../TDVP_lib_VecRho.jl")

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
    eps = parameters["sys_en"]
    U = parameters["spin_interaction"]

    # Input: chain stub parameters
    # ----------------------------
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    empty_chain_coups = thermofield_coefficients[:, 1]
    empty_chain_freqs = thermofield_coefficients[:, 3]
    filled_chain_coups = thermofield_coefficients[:, 2]
    filled_chain_freqs = thermofield_coefficients[:, 4]
    chain_length = parameters["chain_length"]

    # Site ranges
    system_site = 1
    empty_chain = range(; start=2, step=2, length=chain_length)
    filled_chain = range(; start=3, step=2, length=chain_length)

    total_size = system_length + 2chain_length

    sites = siteinds("vElectron", total_size)
    initialstates = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "UpDn" for st in filled_chain]
            [st => "Emp" for st in empty_chain]
        ],
    )
    initialops = Dict(
        [
            system_site => "v"*parameters["system_observable"]
            [st => "vId" for st in filled_chain]
            [st => "vId" for st in empty_chain]
        ],
    )
    init_state = MPS(sites, [initialstates[i] for i in 1:total_size])
    init_targetop = MPS(sites, [initialops[i] for i in 1:total_size])
    opgrade = "even"

    # Unitary part of master equation
    # -------------------------------
    adjℓ = OpSum()

    # System Hamiltonian
    adjℓ += -eps * gkslcommutator("Ntot", system_site)
    adjℓ += -U * gkslcommutator("NupNdn", system_site)

    if chain_length > 0
        # System-chain interaction:
        # c↑ᵢ† c↑ᵢ₊₁ + c↑ᵢ₊₁† c↑ᵢ + c↓ᵢ† c↓ᵢ₊₁ + c↓ᵢ₊₁† c↓ᵢ =
        # a↑ᵢ† Fᵢ a↑ᵢ₊₁ - a↑ᵢ Fᵢ a↑ᵢ₊₁† + a↓ᵢ† Fᵢ₊₁ a↓ᵢ₊₁ - a↓ᵢ Fᵢ₊₁ a↓ᵢ₊₁†
        adjℓ +=
            -empty_chain_coups[1] *
            gkslcommutator("Aup†F", system_site, "Aup", empty_chain[1])
        adjℓ -=
            -empty_chain_coups[1] *
            gkslcommutator("AupF", system_site, "Aup†", empty_chain[1])
        adjℓ +=
            -empty_chain_coups[1] *
            gkslcommutator("Adn†", system_site, "FAdn", empty_chain[1])
        adjℓ -=
            -empty_chain_coups[1] *
            gkslcommutator("Adn", system_site, "FAdn†", empty_chain[1])

            adjℓ += -filled_chain_coups[1] * gkslcommutator("Aup†F",system_site, "F", empty_chain[1], "Aup", filled_chain[1])
            adjℓ -= -filled_chain_coups[1] * gkslcommutator("AupF", system_site, "F", empty_chain[1], "Aup†", filled_chain[1])
            adjℓ += -filled_chain_coups[1] * gkslcommutator("Adn†", system_site, "F", empty_chain[1], "FAdn", filled_chain[1])
            adjℓ -= -filled_chain_coups[1] * gkslcommutator("Adn",  system_site, "F", empty_chain[1], "FAdn†", filled_chain[1])

        # Hamiltonian of the chain stubs:
        for (j, site) in enumerate(empty_chain)
            adjℓ += -empty_chain_freqs[j] * gkslcommutator("Ntot", site)
        end
        for (j, (site1, site2)) in enumerate(partition(empty_chain, 2, 1))
            adjℓ += -empty_chain_coups[j + 1] * gkslcommutator("Aup†F",site1, "F", site1+1, "Aup", site2)
            adjℓ -= -empty_chain_coups[j + 1] * gkslcommutator("AupF", site1, "F", site1+1, "Aup†", site2)
            adjℓ += -empty_chain_coups[j + 1] * gkslcommutator("Adn†", site1, "F", site1+1, "FAdn", site2)
            adjℓ -= -empty_chain_coups[j + 1] * gkslcommutator("Adn",  site1, "F", site1+1, "FAdn†", site2)
        end

        for (j, site) in enumerate(filled_chain)
            adjℓ += -filled_chain_freqs[j] * gkslcommutator("Ntot", site)
        end
        for (j, (site1, site2)) in enumerate(partition(filled_chain, 2, 1))
            adjℓ += -filled_chain_coups[j + 1] * gkslcommutator("Aup†F",site1, "F", site1+1, "Aup", site2)
            adjℓ -= -filled_chain_coups[j + 1] * gkslcommutator("AupF", site1, "F", site1+1, "Aup†", site2)
            adjℓ += -filled_chain_coups[j + 1] * gkslcommutator("Adn†", site1, "F", site1+1, "FAdn", site2)
            adjℓ -= -filled_chain_coups[j + 1] * gkslcommutator("Adn",  site1, "F", site1+1, "FAdn†", site2)
        end
    end

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
