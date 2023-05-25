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
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    empty_Ω = parameters["empty_asympt_frequency"]
    empty_K = parameters["empty_asympt_coupling"]
    filled_Ω = parameters["filled_asympt_frequency"]
    filled_K = parameters["filled_asympt_coupling"]

    α = readdlm(parameters["MC_alphas"])
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    emptymc = closure(empty_Ω, empty_K, α, β, w)
    filledmc = closure(filled_Ω, filled_K, α, β, w)
    closure_length = length(emptymc)

    total_size = system_length + 2chain_length + 2closure_length

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    empty_closure_range = range(; start=empty_chain_range[end] + 2, step=2, length=closure_length)
    filled_chain_range = range(; start=3, step=2, length=chain_length)
    filled_closure_range = range(;
        start=filled_chain_range[end] + 2, step=2, length=closure_length
    )
    @assert filled_closure_range[end] == total_size
    chain_edge_site = max(filled_closure_range..., empty_closure_range...)

    sites = siteinds("vS=1/2", total_size)
    initialsites = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "Up" for st in filled_chain_range]
            [st => "Up" for st in filled_closure_range]
            [st => "Dn" for st in empty_chain_range]
            [st => "Dn" for st in empty_closure_range]
        ],
    )
    psi0 = MPS(sites, [initialsites[i] for i in 1:total_size])

    ℓ = OpSum()

    ℓ += eps * gkslcommutator("N", system_site)

    ℓ +=
        empty_chain_coups[1] * exchange_interaction(
                                                    SiteType("vS=1/2"), sites[system_site], sites[empty_chain_range[1]]
        )
    ℓ +=
        filled_chain_coups[1] * exchange_interaction(
                                                     SiteType("vS=1/2"), sites[system_site], sites[filled_chain_range[1]]
        )

    ℓ += spin_chain(
        SiteType("vS=1/2"), empty_chain_freqs, empty_chain_coups, sites[empty_chain_range]
    )
    ℓ += spin_chain(
        SiteType("vS=1/2"),
        filled_chain_freqs,
        filled_chain_coups,
        sites[filled_chain_range],
    )

    ℓ += closure_op(
        SiteType("vS=1/2"), emptymc, sites[empty_closure_range], chain_edge_site
    )
    ℓ += filled_closure_op(
        SiteType("vS=1/2"), filledmc, sites[filled_closure_range], chain_edge_site
    )

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
