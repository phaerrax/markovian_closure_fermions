using ITensors, ITensorMPS
using HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using TimeEvoVecMPS

let
    parameters = load_pars(ARGS[1])

    # Load initial state from disk
    initstate_file = get(parameters, "initial_state_file", nothing)
    psi_0 = h5open(initstate_file, "r") do file
        return read(file, parameters["initial_state_label"], MPS)
    end
    if parameters["max_bond"] > maxlinkdim(psi_0)
        growMPS!(psi_0, parameters["max_bond"])
    end
    sites = siteinds(psi_0)

    # Input: system parameters
    system_length = 1
    eps = parameters["sys_en"]

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
    total_size != length(psi_0) && error(
        "Number of sites between initial state from file and parameters does not match."
    )

    h = OpSum()
    h += eps, "n", systempos

    h +=
        emptycoups[1] * exchange_interaction(sites[systempos], sites[emptychain_sites[1]]) +
        filledcoups[1] * exchange_interaction(sites[systempos], sites[filledchain_sites[1]])
    +
    spin_chain(
        emptyfreqs[1:chain_length], emptycoups[2:chain_length], sites[emptychain_sites]
    ) + spin_chain(
        filledfreqs[1:chain_length], filledcoups[2:chain_length], sites[filledchain_sites]
    )

    H = MPO(h, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    d = LocalOperator[LocalOperator(Dict(1 => "a"))]
    cb = ExpValueCallback(d, sites, parameters["ms_stride"] * timestep)

    # G_R(t) = -i ⟨ψ| {d(t), d*(0)} |ψ⟩ =
    #        = -i ⟨ψ| d(t) d* |ψ⟩ -i ⟨ψ| d* d(t) |ψ⟩ =
    #        = -i ⟨ψ| U*(t) d U(t) d† |ψ⟩ -i ⟨ψ| d* U*(t) d U(t) |ψ⟩
    # We compute the second term -i ⟨ψ| d* U*(t) d U(t) |ψ⟩
    #                               +----------+   +------+
    #                                  psiL_t       psiR_t
    # that can be later obtained by looking at the "a{1}_im" column of the output file.
    psiL_t = apply(op("a", sites, systempos), psi_0)
    psiR_t = psi_0

    jointtdvp1!(
        (psiL_t, psiR_t),
        H,
        timestep,
        tmax;
        hermitian=true,
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

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_stateL", psiL_t)
            write(f, "final_stateR", psiR_t)
        end
    end
end
