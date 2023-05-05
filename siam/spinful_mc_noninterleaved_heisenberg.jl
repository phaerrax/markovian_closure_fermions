using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
using IterTools

include("../TDVP_lib_VecRho.jl")

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

    # Input: closure parameters
    # -------------------------
    empty_Ω = parameters["empty_asympt_frequency"]
    empty_K = parameters["empty_asympt_coupling"]
    filled_Ω = parameters["filled_asympt_frequency"]
    filled_K = parameters["filled_asympt_coupling"]

    α = readdlm(parameters["MC_alphas"]) # α[l,1] = Re(αₗ) and α[l,2] = Im(αₗ)...
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    empty_closure_ω = @. empty_Ω - 2empty_K * α[:, 2]
    empty_closure_γ = @. -4empty_K * α[:, 1]
    empty_closure_g = @. -2empty_K * β[:, 2]
    empty_closure_ζ = @. empty_K * (w[:, 1] + im * w[:, 2])

    filled_closure_ω = @. filled_Ω - 2filled_K * α[:, 2]
    filled_closure_γ = @. -4filled_K * α[:, 1]
    filled_closure_g = @. -2filled_K * β[:, 2]
    filled_closure_ζ = @. filled_K * (w[:, 1] + im * w[:, 2])

    closure_length = length(empty_closure_ω)

    sites = siteinds("vElectron", system_length + 2chain_length + 2closure_length)
    init_state = MPS(
        sites,
        [
            repeat(["UpDn"], chain_length + closure_length)
            system_initstate
            repeat(["Emp"], chain_length + closure_length)
        ],
    )
    init_targetop = MPS(
        ComplexF64,
        sites,
        [
            repeat(["vId"], chain_length + closure_length)
            "vNtot"
            repeat(["vId"], chain_length + closure_length)
        ],
    )
    opgrade = "even"

    # Site ranges
    system_site = chain_length + closure_length + 1
    filled_chain = (system_site - 1):-1:(system_site - chain_length)
    filled_closure = (system_site - chain_length - 1):-1:1
    empty_chain = system_site .+ (1:chain_length)
    empty_closure = empty_chain[end] .+ (1:closure_length)

    # Unitary part of master equation
    # -------------------------------
    adjℓ = OpSum()

    # System Hamiltonian
    adjℓ += -eps * gkslcommutator("Ntot", system_site)

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

    # Hamiltonian of the closures:
    for (j, site) in enumerate(empty_closure)
        adjℓ += -empty_closure_ω[j] * gkslcommutator("Ntot", site)
    end
    for (j, (site1, site2)) in enumerate(partition(empty_closure, 2, 1))
        adjℓ += -empty_closure_g[j] * gkslcommutator("Aup†F", site1, "Aup", site2)
        adjℓ -= -empty_closure_g[j] * gkslcommutator("AupF", site1, "Aup†", site2)
        adjℓ += -empty_closure_g[j] * gkslcommutator("Adn†", site1, "FAdn", site2)
        adjℓ -= -empty_closure_g[j] * gkslcommutator("Adn", site1, "FAdn†", site2)
    end
    for (j, site) in enumerate(empty_closure)
        # c↑ᵢ† c↑ᵢ₊ₙ = a↑ᵢ† Fᵢ Fᵢ₊₁ ⋯ Fᵢ₊ₙ₋₁ a↑ᵢ₊ₙ
        # c↑ᵢ₊ₙ† c↑ᵢ = -a↑ᵢ Fᵢ Fᵢ₊₁ ⋯ Fᵢ₊ₙ₋₁ a↑ᵢ₊ₙ†
        # c↓ᵢ† c↓ᵢ₊ₙ = a↓ᵢ† Fᵢ₊₁ Fᵢ₊₂ ⋯ Fᵢ₊ₙ a↓ᵢ₊ₙ
        # c↓ᵢ₊ₙ† c↓ᵢ = -a↓ᵢ Fᵢ₊₁ Fᵢ₊₂ ⋯ Fᵢ₊ₙ a↓ᵢ₊ₙ†
        chainedge = empty_chain[end]
        ps_length = abs(site - chainedge) - 1
        ps_range = chainedge > site ? (chainedge:-1:site) : (chainedge:site)

        # ζⱼ c↑₀† c↑ⱼ (0 = chain edge, j = pseudomode)
        adjℓ +=
            -empty_closure_ζ[j] *
            gkslcommutator(zip(["Aup†F"; repeat(["F"], ps_length); "Aup"], ps_range)...)
        # conj(ζⱼ) c↑ⱼ† c↑₀
        adjℓ -=
            -conj(empty_closure_ζ[j]) *
            gkslcommutator(zip(["AupF"; repeat(["F"], ps_length); "Aup†"], ps_range)...)
        # ζⱼ c↓₀† c↓ⱼ (0 = chain edge, j = pseudomode)
        adjℓ +=
            -empty_closure_ζ[j] *
            gkslcommutator(zip(["Adn†"; repeat(["F"], ps_length); "FAdn"], ps_range)...)
        # conj(ζⱼ) c↓ⱼ† c↓₀
        adjℓ -=
            -conj(empty_closure_ζ[j]) *
            gkslcommutator(zip(["Adn"; repeat(["F"], ps_length); "FAdn†"], ps_range)...)
    end

    for (j, site) in enumerate(filled_closure)
        adjℓ += -filled_closure_ω[j] * gkslcommutator("Ntot", site)
    end
    for (j, (site1, site2)) in enumerate(partition(filled_closure, 2, 1))
        adjℓ += -filled_closure_g[j] * gkslcommutator("Aup†F", site1, "Aup", site2)
        adjℓ -= -filled_closure_g[j] * gkslcommutator("AupF", site1, "Aup†", site2)
        adjℓ += -filled_closure_g[j] * gkslcommutator("Adn†", site1, "FAdn", site2)
        adjℓ -= -filled_closure_g[j] * gkslcommutator("Adn", site1, "FAdn†", site2)
    end
    for (j, site) in enumerate(filled_closure)
        chainedge = filled_chain[end]
        ps_length = abs(site - chainedge) - 1
        ps_range = chainedge > site ? (chainedge:-1:site) : (chainedge:site)

        # ζⱼ c↑₀† c↑ⱼ (0 = chain edge, j = pseudomode)
        adjℓ +=
            -filled_closure_ζ[j] *
            gkslcommutator(zip(["Aup†F"; repeat(["F"], ps_length); "Aup"], ps_range)...)
        # conj(ζⱼ) c↑ⱼ† c↑₀
        adjℓ -=
            -conj(filled_closure_ζ[j]) *
            gkslcommutator(zip(["AupF"; repeat(["F"], ps_length); "Aup†"], ps_range)...)
        # ζⱼ c↓₀† c↓ⱼ (0 = chain edge, j = pseudomode)
        adjℓ +=
            -filled_closure_ζ[j] *
            gkslcommutator(zip(["Adn†"; repeat(["F"], ps_length); "FAdn"], ps_range)...)
        # conj(ζⱼ) c↓ⱼ† c↓₀
        adjℓ -=
            -conj(filled_closure_ζ[j]) *
            gkslcommutator(zip(["Adn"; repeat(["F"], ps_length); "FAdn†"], ps_range)...)
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    # The ±c†Xc term in the GKSL equation changes sign according to the grade (parity) of
    # the operator to be evolved. The rest is as usual.
    gradefactor = opgrade == "even" ? 1 : -1
    for (j, site) in enumerate(empty_closure)
        # Remember that:
        # • Fⱼ = (1 - 2 N↑ₖ) (1 - 2 N↓ₖ);
        # • Fⱼ and aₛ,ₖ commute only on different sites;
        # • {a↓ₖ, a↓ₖ†} = {a↑ₖ, a↑ₖ†} = 1;
        # • Fₖ anticommutes with a↓ₖ, a↓ₖ†, a↑ₖ and a↑ₖ†.

        # c↑ₖ† ρ c↑ₖ = a↑ₖ† Fₖ₋₁ ⋯ F₁ ρ F₁ ⋯ Fₖ₋₁ a↑ₖ
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Aup†⋅ * ⋅Aup"]
        adjℓ += (
            gradefactor * empty_closure_γ[j],
            collect(Iterators.flatten(zip(opstring, 1:site)))...,
        )
        # c↓ₖ† ρ c↓ₖ = a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁ ρ F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Adn†F⋅ * ⋅FAdn"]
        adjℓ += (
            gradefactor * empty_closure_γ[j],
            collect(Iterators.flatten(zip(opstring, 1:site)))...,
        )

        # -½ (c↑ₖ† c↑ₖ ρ + ρ c↑ₖ† c↑ₖ) = -½ (a↑ₖ† a↑ₖ ρ + ρ a↑ₖ† a↑ₖ)
        adjℓ += -0.5empty_closure_γ[j], "Nup⋅", site
        adjℓ += -0.5empty_closure_γ[j], "⋅Nup", site
        # -½ (c↓ₖ† c↓ₖ ρ + ρ c↓ₖ† c↓ₖ) = -½ (a↓ₖ† a↓ₖ ρ + ρ a↓ₖ† a↓ₖ)
        adjℓ += -0.5empty_closure_γ[j], "Ndn⋅", site
        adjℓ += -0.5empty_closure_γ[j], "⋅Ndn", site
    end
    for (j, site) in enumerate(filled_closure)
        # c↑ₖ ρ c↑ₖ† = F₁ ⋯ Fₖ₋₁ a↑ₖ ρ a↑ₖ† Fₖ₋₁ ⋯ F₁
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Aup⋅ * ⋅Aup†"]
        adjℓ += (
            gradefactor * filled_closure_γ[j],
            collect(Iterators.flatten(zip(opstring, 1:site)))...,
        )
        # c↓ₖ ρ c↓ₖ† = F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ ρ a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "FAdn⋅ * ⋅Adn†F"]
        adjℓ += (
            gradefactor * filled_closure_γ[j],
            collect(Iterators.flatten(zip(opstring, 1:site)))...,
        )

        # c↑ₖ c↑ₖ† = F₁ ⋯ Fₖ₋₁ a↑ₖ a↑ₖ† Fₖ₋₁ ⋯ F₁ =
        #          = F₁² ⋯ Fₖ₋₁² a↑ₖ a↑ₖ† =
        #          = a↑ₖ a↑ₖ† =
        #          = 1 - N↑ₖ
        adjℓ += -filled_closure_γ[j], "Id", site
        adjℓ += 0.5filled_closure_γ[j], "Nup⋅", site
        adjℓ += 0.5filled_closure_γ[j], "⋅Nup", site
        # c↓ₖ c↓ₖ† = F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁ =
        #          = F₁² ⋯ Fₖ₋₁² Fₖa↓ₖ a↓ₖ†Fₖ =
        #          = Fₖa↓ₖ a↓ₖ†Fₖ =
        #          = Fₖ² a↓ₖ a↓ₖ† =
        #          = 1 - N↓ₖ
        adjℓ += -filled_closure_γ[j], "Id", site
        adjℓ += 0.5filled_closure_γ[j], "Ndn⋅", site
        adjℓ += 0.5filled_closure_γ[j], "⋅Ndn", site
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
