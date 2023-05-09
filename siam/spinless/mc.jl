using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
using IterTools

# This script tries to emulate the simulation of the non-interacting SIAM model
# described in Lucas Kohn's PhD thesis (section 4.2.1).
# An impurity is interacting with a fermionic thermal bath, which is mapped onto two
# discrete chains by means of a thermofield+TEDOPA transformation.
# The Markovian closure technique is then applied to _both_ chains, truncating the two
# environments and replacing part of them with sets of pseudomodes.
# The two chains are then interleaved so as to build a single chain. This is hardcoded,
# for now; maybe sometime I'll find a way to calculate the Jordan-Wigner operators in
# an automatic way...

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

    chain_length = parameters["chain_length"]
    closure_length = length(empty_closure_ω)
    total_size = system_length + 2chain_length + 2closure_length

    # Site ranges
    system_site = 1
    filled_chain = range(; start=3, step=2, length=chain_length)
    filled_closure = range(; start=filled_chain[end] + 2, step=2, length=closure_length)
    empty_chain = range(; start=2, step=2, length=chain_length)
    empty_closure = range(; start=empty_chain[end] + 2, step=2, length=closure_length)

    sites = siteinds("vS=1/2", total_size)
    initialsites = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "Up" for st in filled_chain]
            [st => "Up" for st in filled_closure]
            [st => "Dn" for st in empty_chain]
            [st => "Dn" for st in empty_closure]
        ],
    )
    psi0 = MPS(sites, [initialsites[i] for i in 1:total_size])

    # Unitary part of master equation
    # -------------------------------
    # Note that some two-site operators on the filled part of the system pick up an
    # additional minus sign, due to the Jordan-Wigner transformation.
    ℓ = OpSum()

    # System Hamiltonian
    ℓ += eps * gkslcommutator("N", system_site)

    if chain_length > 0
        # System-chain interaction:
        ℓ += empty_chain_coups[1] * gkslcommutator("σ+", system_site, "σ-", empty_chain[1])
        ℓ += empty_chain_coups[1] * gkslcommutator("σ-", system_site, "σ+", empty_chain[1])
        ℓ +=
            -filled_chain_coups[1] *
            gkslcommutator("σ+", system_site, "σz", empty_chain[1], "σ-", filled_chain[1])
        ℓ +=
            -filled_chain_coups[1] *
            gkslcommutator("σ-", system_site, "σz", empty_chain[1], "σ+", filled_chain[1])

        # Hamiltonian of the chain stubs:
        for (j, site) in enumerate(empty_chain)
            ℓ += empty_chain_freqs[j] * gkslcommutator("N", site)
        end
        for (j, (site1, site2)) in enumerate(partition(empty_chain, 2, 1))
            ℓ +=
                -empty_chain_coups[j + 1] *
                gkslcommutator("σ+", site1, "σz", site1 + 1, "σ-", site2)
            ℓ +=
                -empty_chain_coups[j + 1] *
                gkslcommutator("σ-", site1, "σz", site1 + 1, "σ+", site2)
        end

        for (j, site) in enumerate(filled_chain)
            ℓ += filled_chain_freqs[j] * gkslcommutator("N", site)
        end
        for (j, (site1, site2)) in enumerate(partition(filled_chain, 2, 1))
            ℓ +=
                -filled_chain_coups[j + 1] *
                gkslcommutator("σ+", site1, "σz", site1 + 1, "σ-", site2)
            ℓ +=
                -filled_chain_coups[j + 1] *
                gkslcommutator("σ-", site1, "σz", site1 + 1, "σ+", site2)
        end
    end

    # Hamiltonian of the closures:
    for (j, site) in enumerate(empty_closure)
        ℓ += empty_closure_ω[j] * gkslcommutator("N", site)
    end
    for (j, (site1, site2)) in enumerate(partition(empty_closure, 2, 1))
        ℓ += -empty_closure_g[j] * gkslcommutator("σ-", site1, "σz", site1 + 1, "σ+", site2)
        ℓ += -empty_closure_g[j] * gkslcommutator("σ+", site1, "σz", site1 + 1, "σ-", site2)
    end
    for (j, site) in enumerate(empty_closure)
        chainedge = empty_chain[end]
        ps_length = abs(site - chainedge) - 1
        ps_range = chainedge > site ? (chainedge:-1:site) : (chainedge:site)

        paulistring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        ℓ +=
            (-1)^ps_length *
            empty_closure_ζ[j] *
            gkslcommutator(zip(paulistring, ps_range)...)

        paulistring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        ℓ +=
            (-1)^ps_length *
            conj(empty_closure_ζ[j]) *
            gkslcommutator(zip(paulistring, ps_range)...)
    end

    for (j, site) in enumerate(filled_closure)
        ℓ += filled_closure_ω[j] * gkslcommutator("N", site)
    end
    for (j, (site1, site2)) in enumerate(partition(filled_closure, 2, 1))
        ℓ +=
            -filled_closure_g[j] * gkslcommutator("σ-", site1, "σz", site1 + 1, "σ+", site2)
        ℓ +=
            -filled_closure_g[j] * gkslcommutator("σ+", site1, "σz", site1 + 1, "σ-", site2)
    end
    for (j, site) in enumerate(filled_closure)
        chainedge = filled_chain[end]
        ps_length = abs(site - chainedge) - 1
        ps_range = chainedge > site ? (chainedge:-1:site) : (chainedge:site)

        paulistring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        ℓ +=
            (-1)^ps_length *
            filled_closure_ζ[j] *
            gkslcommutator(zip(paulistring, ps_range)...)

        paulistring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        ℓ +=
            (-1)^ps_length *
            conj(filled_closure_ζ[j]) *
            gkslcommutator(zip(paulistring, ps_range)...)
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    @debug "Empty closure modes"
    for (j, site) in enumerate(empty_closure)
        # a ρ a†
        opstring = [repeat(["σz⋅ * ⋅σz"], site - 1); "σ-⋅ * ⋅σ+"]
        @show opstring
        @show collect(Iterators.flatten(zip(opstring, 1:site)))
        ℓ += (empty_closure_γ[j], collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ += -0.5empty_closure_γ[j], "N⋅", site
        ℓ += -0.5empty_closure_γ[j], "⋅N", site
    end
    @debug "Filled closure modes"
    for (j, site) in enumerate(filled_closure)
        # a ρ a†
        opstring = [repeat(["σz⋅ * ⋅σz"], site - 1); "σ+⋅ * ⋅σ-"]
        @show opstring
        @show collect(Iterators.flatten(zip(opstring, 1:site)))
        ℓ += (filled_closure_γ[j], collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ += 0.5filled_closure_γ[j], "N⋅", site
        ℓ += 0.5filled_closure_γ[j], "⋅N", site
        ℓ += -filled_closure_γ[j], "Id", site
    end
    L = MPO(ℓ, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosVecMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        psi, _ = stretchBondDim(psi0, parameters["max_bond"])
        tdvp1vec!(
            psi,
            L,
            timestep,
            tmax,
            sites;
            hermitian=false,
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
        psi, _ = stretchBondDim(psi0, 2)
        adaptivetdvp1vec!(
            psi,
            L,
            timestep,
            tmax,
            sites;
            hermitian=false,
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
