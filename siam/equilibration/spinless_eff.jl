using ITensors
using IterTools
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
import KrylovKit: exponentiate

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_length = 1
    ε = parameters["sys_en"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]
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

    # Site ranges
    system_site = 1
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    empty_closure_range = range(;
        start=empty_chain_range[end] + 2, step=2, length=closure_length
    )
    filled_chain_range = range(; start=3, step=2, length=chain_length)
    filled_closure_range = range(;
        start=filled_chain_range[end] + 2, step=2, length=closure_length
    )

    total_size = system_length + 2chain_length + 2closure_length

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            system_site => parameters["sys_ini"]
            [st => "Occ" for st in filled_chain_range]
            [st => "Occ" for st in filled_closure_range]
            [st => "Emp" for st in empty_chain_range]
            [st => "Emp" for st in empty_closure_range]
        ],
    )
    psi0 = MPS(sites, [initialsites[i] for i in 1:total_size])

    h_loc = OpSum()

    h_loc += ε, "n", system_site

    H_lochyb = MPO(
        h_loc +
        emptycoups[1] *
        exchange_interaction(sites[system_site], sites[empty_chain_range[1]]) +
        filledcoups[1] *
        exchange_interaction(sites[system_site], sites[filled_chain_range[1]]),
        sites,
    )

    h_cond = OpSum()
    h_cond += spin_chain(
        emptyfreqs[1:chain_length], emptycoups[2:chain_length], sites[empty_chain_range]
    )
    h_cond += spin_chain(
        filledfreqs[1:chain_length], filledcoups[2:chain_length], sites[filled_chain_range]
    )

    h_effclosure = OpSum()
    h_effclosure += spin_chain(
        freqs(emptymc), innercoups(emptymc), sites[empty_closure_range]
    )
    h_effclosure += spin_chain(
        freqs(filledmc), innercoups(filledmc), sites[filled_closure_range]
    )
    for (i, site) in enumerate(empty_closure_range)
        h_effclosure += outercoup(emptymc, i), "c†", empty_chain_range[end], "c", site
        h_effclosure += conj(outercoup(emptymc, i)), "c†", site, "c", empty_chain_range[end]
        h_effclosure += -0.5im * damp(emptymc, i), "n", site
    end
    for (i, site) in enumerate(filled_closure_range)
        h_effclosure += outercoup(filledmc, i), "c†", filled_chain_range[end], "c", site
        h_effclosure += (
            conj(outercoup(filledmc, i)), "c†", site, "c", filled_chain_range[end]
        )
        h_effclosure += -0.5im * damp(filledmc, i), "c * c†", site
    end
    H_effcond = MPO(h_cond + h_effclosure, sites)

    slope = parameters["adiabatic_ramp_slope"]
    ramp(t) = slope * t < 1 ? convert(typeof(t), slope * t) : one(t)
    fs = [ramp, one]
    Hs = [H_lochyb, H_effcond]

    krylov_kwargs = (;
        ishermitian=false,
        krylovdim=parameters["krylov_dim"],
        tol=parameters["exp_tol"],
        issymmetric=false,
    )

    #  Specific solver function for time-dependent TDVP.
    function time_dependent_solver(
        H::TimeDependentSum, time_step, ψ₀; current_time=0.0, outputlevel=0, kwargs...
    )
        @debug "In Krylov solver, current_time = $current_time, time_step = $time_step"
        ψₜ, info = exponentiate(H(current_time), time_step, ψ₀; kwargs...)
        return ψₜ, info
    end

    function td_solver(Hs::ProjMPOSum, time_step, ψ₀; kwargs...)
        return time_dependent_solver(
            TimeDependentSum(fs, Hs), time_step, ψ₀; krylov_kwargs..., kwargs...
        )
    end

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

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        psi, _ = stretchBondDim(psi0, parameters["max_bond"])
        tdvp1!(
            td_solver,
            psi,
            Hs,
            timestep,
            tmax;
            normalize=false,
            hermitian=false,
            callback=cb,
            progress=true,
            store_psi0=false,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    else
        @info "Using adaptive algorithm."
        psi, _ = stretchBondDim(psi0, 2)
        adaptivetdvp1!(
            td_solver,
            psi,
            Hs,
            timestep,
            tmax;
            normalize=false,
            callback=cb,
            progress=true,
            store_psi0=false,
            hermitian=false,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
        )
    end
end
