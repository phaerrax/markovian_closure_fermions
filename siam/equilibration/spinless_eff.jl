using ITensors
using ITensors.HDF5
using IterTools
using DelimitedFiles
using LindbladVectorizedTensors
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

    α_mat = readdlm(parameters["MC_alphas"])
    β_mat = readdlm(parameters["MC_betas"])
    w_mat = readdlm(parameters["MC_coups"])

    α = α_mat[:, 1] .+ im .* α_mat[:, 2]
    β = β_mat[:, 1] .+ im .* β_mat[:, 2]
    w = w_mat[:, 1] .+ im .* w_mat[:, 2]

    emptymc = closure(empty_Ω, empty_K, α, β, w)
    filledmc = closure(filled_Ω, filled_K, conj.(α), conj.(β), w)
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

    operators = LocalOperator[]
    for (k, v) in parameters["observables"]
        for n in v
            push!(operators, LocalOperator(Dict(n => k)))
        end
    end
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    psi, _ = stretchBondDim(psi0, parameters["max_bond"])
    intermediate_state_file = append_if_not_null(parameters["state_file"], "_intermediate")

    @info "Starting equilibration algorithm."
    tdvp1!(
        td_solver,
        psi,
        Hs,
        timestep,
        inv(slope);
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
    write(f, "intermediate_state", psi)
    close(f)

    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    tdvp1!(
        psi,
        H_lochyb + H_effcond,
        timestep,
        tmax - inv(slope);
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

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", psi)
        end
    end
end
