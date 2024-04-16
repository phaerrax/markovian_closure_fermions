using ITensors
using ITensors.HDF5
using DelimitedFiles
using LindbladVectorizedTensors
using TimeEvoVecMPS

function dot_hamiltonian(
    ::SiteType"Electron", dot_energies, dot_coulomb_repulsion, dot_site
)
    # 1st level --> spin ↓
    # 2nd level --> spin ↑
    h = OpSum()

    h += dot_energies[1], "Ndn", dot_site
    h += dot_energies[2], "Nup", dot_site

    h += 0.5dot_coulomb_repulsion, "Ntot^2", dot_site
    h += -0.5dot_coulomb_repulsion, "Ntot", dot_site

    return h
end

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    dot_energies = parameters["dot_energies"]
    dot_coulomb_repulsion = parameters["dot_coulomb_repulsion"]

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
    total_size = 1 + 2chain_length

    # Site ranges
    dot_site = 1
    empty_chain_range = range(; start=2, step=2, length=chain_length)
    filled_chain_range = range(; start=3, step=2, length=chain_length)
    @assert filled_chain_range[end] == total_size

    initstate_file = get(parameters, "initial_state_file", nothing)
    if isnothing(initstate_file)
        sites = siteinds(n -> n == 1 ? "Electron" : "Fermion", total_size)
        initialsites = Dict(
            [
                dot_site => "Emp"
                [st => "Occ" for st in filled_chain_range]
                [st => "Emp" for st in empty_chain_range]
            ],
        )
        ψₜ = MPS(sites, [initialsites[i] for i in 1:total_size])
        start_from_file = false
    else
        ψₜ = h5open(initstate_file, "r") do file
            return read(file, parameters["initial_state_label"], MPS)
        end
        sites = siteinds(ψₜ)
        start_from_file = true
        # We need to extract the site indices from ψₜ or else, if we define them from
        # scratch, they will have different IDs and they won't contract correctly.
    end

    H = MPO(
        dot_hamiltonian(
            SiteType("Electron"), dot_energies, dot_coulomb_repulsion, dot_site
        ) +
        exchange_interaction(
            sites[dot_site],
            sites[empty_chain_range[1]];
            coupling_constant_up=empty_chain_coups[1],
            coupling_constant_dn=empty_chain_coups[1],
        ) +
        exchange_interaction(
            sites[dot_site],
            sites[filled_chain_range[1]];
            coupling_constant_up=filled_chain_coups[1],
            coupling_constant_dn=filled_chain_coups[1],
        ) +
        spin_chain(
            empty_chain_freqs[1:chain_length],
            empty_chain_coups[2:chain_length],
            sites[empty_chain_range],
        ) +
        spin_chain(
            filled_chain_freqs[1:chain_length],
            filled_chain_coups[2:chain_length],
            sites[filled_chain_range],
        ),
        sites,
    )

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    operators = LocalOperator[]
    for (k, v) in parameters["observables"]
        for n in v
            push!(operators, LocalOperator(Dict(n => k)))
        end
    end
    cb = ExpValueCallback(operators, sites, parameters["ms_stride"] * timestep)

    if get(parameters, "convergence_factor_bondadapt", 0) == 0
        @info "Using standard algorithm."
        if !start_from_file
            growMPS!(ψₜ, parameters["max_bond"])
        end
        tdvp1!(
            ψₜ,
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
    else
        @info "Using adaptive algorithm."
        if !start_from_file
            growMPS!(ψₜ, 2)
        end
        adaptivetdvp1!(
            ψₜ,
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
            convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
            max_bond=parameters["max_bond"],
        )
    end

    if parameters["state_file"] != "/dev/null"
        h5open(parameters["state_file"], "w") do f
            write(f, "final_state", ψₜ)
        end
    end
end
