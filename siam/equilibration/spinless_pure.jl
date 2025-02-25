using ITensors, ITensorMPS
using HDF5
using IterTools
using DelimitedFiles
using LindbladVectorizedTensors
using MPSTimeEvolution
import KrylovKit: exponentiate

include("../shared_functions.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_length = 1
    eps = parameters["sys_en"]
    #U = parameters["spin_interaction"]

    # Input: chain parameters
    thermofield_coefficients = readdlm(
        parameters["chain_coefficients"], ',', Float64; skipstart=1
    )
    emptycoups = thermofield_coefficients[:, 1]
    emptyfreqs = thermofield_coefficients[:, 3]
    filledcoups = thermofield_coefficients[:, 2]
    filledfreqs = thermofield_coefficients[:, 4]

    chain_length = parameters["chain_length"]
    total_size = 2 * chain_length + 1
    systempos = 1
    filledchain_sites = 3:2:total_size
    emptychain_sites = 2:2:total_size

    sites = siteinds("Fermion", total_size)
    initialsites = Dict(
        [
            systempos => parameters["sys_ini"]
            [st => "Occ" for st in filledchain_sites]
            [st => "Emp" for st in emptychain_sites]
        ],
    )
    psi0 = MPS(sites, [initialsites[i] for i in 1:total_size])

    h_loc = OpSum()
    h_loc += eps, "n", systempos

    H_lochyb = MPO(
        h_loc +
        emptycoups[1] * exchange_interaction(sites[systempos], sites[emptychain_sites[1]]) +
        filledcoups[1] * exchange_interaction(sites[systempos], sites[filledchain_sites[1]]),
        sites,
    )

    H_cond = MPO(
        spin_chain(
            emptyfreqs[1:chain_length], emptycoups[2:chain_length], sites[emptychain_sites]
        ) + spin_chain(
            filledfreqs[1:chain_length],
            filledcoups[2:chain_length],
            sites[filledchain_sites],
        ),
        sites,
    )

    slope = parameters["adiabatic_ramp_slope"]
    ramp(t) = slope * t < 1 ? convert(typeof(t), slope * t) : one(t)
    fs = [ramp, one]
    Hs = [H_lochyb, H_cond]

    krylov_kwargs = (;
        ishermitian=true, krylovdim=parameters["krylov_dim"], tol=parameters["exp_tol"]
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
        ishermitian=true,
        issymmetric=false,
        io_file=append_if_not_null(parameters["out_file"], "_ramp"),
        io_ranks=append_if_not_null(parameters["ranks_file"], "_ramp"),
        io_times=append_if_not_null(parameters["times_file"], "_ramp"),
    )

    f = h5open(intermediate_state_file, "w")
    write(f, "intermediate_state", psi)
    close(f)

    # Reset callback object
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    tdvp1!(
        psi,
        H_lochyb + H_cond,
        timestep,
        tmax - inv(slope);
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        ishermitian=true,
        issymmetric=false,
        io_file=append_if_not_null(parameters["out_file"], "_relax"),
        io_ranks=append_if_not_null(parameters["ranks_file"], "_relax"),
        io_times=append_if_not_null(parameters["times_file"], "_relax"),
    )

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", psi)
        end
    end
end
