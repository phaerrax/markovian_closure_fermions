using ITensors
using ITensors.HDF5
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
using IterTools
import KrylovKit: exponentiate

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

    sites = siteinds("vElectron", total_size)
    initstate = MPS(
        sites,
        Dict(
            [
                system_site => parameters["sys_ini"]
                [st => "UpDn" for st in filled_chain_range]
                [st => "UpDn" for st in filled_closure_range]
                [st => "Emp" for st in empty_chain_range]
                [st => "Emp" for st in empty_closure_range]
            ],
        ),
    )

    # The operators are split as in Kohn's PhD dissertation.
    L_lochyb = MPO(
        ε * gkslcommutator("Ntot", system_site) +
        U * gkslcommutator("NupNdn", system_site) +
        empty_chain_coups[1] *
        exchange_interaction(sites[system_site], sites[empty_chain_range[1]]) +
        filled_chain_coups[1] *
        exchange_interaction(sites[system_site], sites[filled_chain_range[1]]),
        sites,
    )
    L_cond = MPO(
        spin_chain(
            empty_chain_freqs[1:chain_length],
            empty_chain_coups[2:chain_length],
            sites[empty_chain_range],
        ) +
        spin_chain(
            filled_chain_freqs[1:chain_length],
            filled_chain_coups[2:chain_length],
            sites[filled_chain_range],
        ) +
        closure_op(emptymc, sites[empty_closure_range], empty_chain_range[end]) +
        filled_closure_op(filledmc, sites[filled_closure_range], filled_chain_range[end]),
        sites,
    )

    slope = parameters["adiabatic_ramp_slope"]
    ramp(t) = t * slope < 1 ? convert(typeof(t), slope * t) : one(t)
    fs = [ramp, one]
    Ls = [L_lochyb, L_cond]

    krylov_kwargs = (;
        ishermitian=false, krylovdim=parameters["krylov_dim"], tol=parameters["exp_tol"]
    )

    #  Specific solver function for time-dependent TDVP.
    function time_dependent_solver(
        L::TimeDependentSum, time_step, vecρ₀; current_time=0.0, outputlevel=0, kwargs...
    )
        @debug "In Krylov solver, current_time = $current_time, time_step = $time_step"
        vecρₜ, info = exponentiate(L(current_time), time_step, vecρ₀; kwargs...)
        return vecρₜ, info
    end

    function td_solver(Ls::ProjMPOSum, time_step, vecρ₀; kwargs...)
        # Questa è la funzione che viene innestata in tdvp_site_update!.
        # Le vengono forniti gli argomenti (H, time_step, psi; current_time); con H costruiamo
        # l'oggetto TimeDependentSum.
        # A sua volta, chiama time_dependent_solver con un TimeDependentSum, che è definito in
        # TimeEvoVecMPS/src/tdvp_step.jl. Qui attacchiamo anche i kwargs specifici di exponentiate,
        # che vengono passati alla funzione dal time_dependent_solver "più interno".
        return time_dependent_solver(
            TimeDependentSum(fs, Ls), time_step, vecρ₀; krylov_kwargs..., kwargs...
        )
    end

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

    vecρ, _ = stretchBondDim(initstate, parameters["max_bond"])
    intermediate_state_file = append_if_not_null(parameters["state_file"], "_intermediate")

    @info "Starting equilibration algorithm."
    tdvp1vec!(
        td_solver,
        vecρ,
        Ls,
        timestep,
        inv(slope),
        sites;
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        ishermitian=false,
        issymmetric=false,
        io_file=append_if_not_null(parameters["out_file"], "_ramp"),
        io_ranks=append_if_not_null(parameters["ranks_file"], "_ramp"),
        io_times=append_if_not_null(parameters["times_file"], "_ramp"),
    )

    f = h5open(intermediate_state_file, "w")
    write(f, "intermediate_state", vecρ)
    close(f)

    cb = LocalPosVecMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    tdvp1vec!(
        vecρ,
        L_lochyb + L_cond,
        timestep,
        tmax - inv(slope),
        sites;
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        ishermitian=false,
        issymmetric=false,
        io_file=append_if_not_null(parameters["out_file"], "_relax"),
        io_ranks=append_if_not_null(parameters["ranks_file"], "_relax"),
        io_times=append_if_not_null(parameters["times_file"], "_relax"),
    )

    f = h5open(parameters["state_file"], "w")
    write(f, "final_state", vecρ)
    close(f)
end
