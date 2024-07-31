using ITensors
using ITensors.HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using MarkovianClosure
using TimeEvoVecMPS
import KrylovKit: exponentiate

let
    config_filename = ARGS[1]
    parameters = load_pars(config_filename)

    # Input: system parameters
    # ------------------------
    system_initstate = parameters["sys_ini"]
    system_length = 1
    ε = parameters["sys_en"]

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

    # Input: closure parameters
    # -------------------------
    empty_Ω = meanordefault(
        empty_chain_freqs[(chain_length + 1):end],
        get(parameters, "empty_asympt_frequency", nothing),
    )
    empty_K = meanordefault(
        empty_chain_coups[(chain_length + 1):end],
        get(parameters, "empty_asympt_coupling", nothing),
    )
    filled_Ω = meanordefault(
        filled_chain_freqs[(chain_length + 1):end],
        get(parameters, "filled_asympt_frequency", nothing),
    )
    filled_K = meanordefault(
        filled_chain_coups[(chain_length + 1):end],
        get(parameters, "filled_asympt_coupling", nothing),
    )

    α_mat = readdlm(parameters["MC_alphas"])
    β_mat = readdlm(parameters["MC_betas"])
    w_mat = readdlm(parameters["MC_coups"])

    α = α_mat[:, 1] .+ im .* α_mat[:, 2]
    β = β_mat[:, 1] .+ im .* β_mat[:, 2]
    w = w_mat[:, 1] .+ im .* w_mat[:, 2]

    emptymc = markovianclosure_parameters(empty_Ω, empty_K, α, β, w)
    filledmc = markovianclosure_parameters(filled_Ω, filled_K, conj.(α), conj.(β), w)
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

    sites = siteinds("vFermion", total_size)
    initstate = MPS(
        sites,
        Dict(
            [
                system_site => parameters["sys_ini"]
                [st => "Up" for st in filled_chain_range]
                [st => "Up" for st in filled_closure_range]
                [st => "Dn" for st in empty_chain_range]
                [st => "Dn" for st in empty_closure_range]
            ],
        ),
    )

    # The operators are split as in Kohn's PhD dissertation.
    L_lochyb = MPO(
        ε * gkslcommutator("N", system_site) +
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
        markovianclosure(emptymc, sites[empty_closure_range], empty_chain_range[end], -1) +
        filled_markovianclosure(
            filledmc, sites[filled_closure_range], filled_chain_range[end], -1
        ),
        sites,
    )

    slope = parameters["adiabatic_ramp_slope"]
    ramp(t) = slope * t < 1 ? convert(typeof(t), slope * t) : one(t)

    fs = [ramp, one]
    Ls = [L_lochyb, L_cond]

    krylov_kwargs = (;
        ishermitian=false,
        krylovdim=parameters["krylov_dim"],
        tol=parameters["exp_tol"],
        issymmetric=false,
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

    operators = LocalOperator[]
    for (k, v) in parameters["observables"]
        for n in v
            push!(operators, LocalOperator(Dict(n => k)))
        end
    end
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

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

    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

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

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", vecρ)
        end
    end
end
