using ITensors
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
    eps = parameters["sys_en"]
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

    sites = siteinds("vElectron", system_length + 2chain_length)
    init_state = MPS(
        sites,
        [
            repeat(["UpDn"], chain_length)
            system_initstate
            repeat(["Emp"], chain_length)
        ],
    )
    init_targetop = MPS(
        #ComplexF64,
        sites,
        [
            repeat(["vId"], chain_length)
            "vNtot"
            repeat(["vId"], chain_length)
        ],
    )
    opgrade = "even"

    # Site ranges
    system_site = chain_length + 1
    filled_chain = (system_site - 1):-1:1
    empty_chain = system_site .+ (1:chain_length)

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

        adjℓ +=
            -filled_chain_coups[1] *
            gkslcommutator("Aup†F", system_site, "Aup", filled_chain[1])
        adjℓ -=
            -filled_chain_coups[1] *
            gkslcommutator("AupF", system_site, "Aup†", filled_chain[1])
        adjℓ +=
            -filled_chain_coups[1] *
            gkslcommutator("Adn†", system_site, "FAdn", filled_chain[1])
        adjℓ -=
            -filled_chain_coups[1] *
            gkslcommutator("Adn", system_site, "FAdn†", filled_chain[1])

        # Hamiltonian of the chain stubs:
        for (j, site) in enumerate(empty_chain)
            adjℓ += -empty_chain_freqs[j] * gkslcommutator("Ntot", site)
        end
        for (j, (site1, site2)) in enumerate(partition(empty_chain, 2, 1))
            adjℓ += -empty_chain_coups[j + 1] * gkslcommutator("Aup†F", site1, "Aup", site2)
            adjℓ -= -empty_chain_coups[j + 1] * gkslcommutator("AupF", site1, "Aup†", site2)
            adjℓ += -empty_chain_coups[j + 1] * gkslcommutator("Adn†", site1, "FAdn", site2)
            adjℓ -= -empty_chain_coups[j + 1] * gkslcommutator("Adn", site1, "FAdn†", site2)
        end

        for (j, site) in enumerate(filled_chain)
            adjℓ += -filled_chain_freqs[j] * gkslcommutator("Ntot", site)
        end
        for (j, (site1, site2)) in enumerate(partition(filled_chain, 2, 1))
            adjℓ +=
                -filled_chain_coups[j + 1] * gkslcommutator("Aup†F", site1, "Aup", site2)
            adjℓ -=
                -filled_chain_coups[j + 1] * gkslcommutator("AupF", site1, "Aup†", site2)
            adjℓ +=
                -filled_chain_coups[j + 1] * gkslcommutator("Adn†", site1, "FAdn", site2)
            adjℓ -=
                -filled_chain_coups[j + 1] * gkslcommutator("Adn", site1, "FAdn†", site2)
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
