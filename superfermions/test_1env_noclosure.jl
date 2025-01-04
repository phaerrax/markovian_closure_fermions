using ITensors, ITensorMPS, MPSTimeEvolution, ArgParse, MKL, CSV

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcoupling,
    nenvironment,
    environment_chain_frequencies,
    environment_chain_couplings,
    dt,
    tmax,
    maxbonddim,
    io_file=nullfile(),
    io_ranks=nullfile(),
    io_times=nullfile(),
    operators,
)
    sites = siteinds("Fermion", 2nsystem + 2nenvironment; conserve_nfparity=true)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [system_energy], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [system_energy], [])

    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=nenvironment),
        environment_chain_frequencies,
        environment_chain_couplings,
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=nenvironment),
        environment_chain_frequencies,
        environment_chain_couplings,
    )

    initstate = MPS(sites, n -> n ≤ 2nsystem ? "Occ" : "Emp")

    ad_h =
        spinchain(SiteType("Fermion"), join(system, environment, sysenvcoupling)) -
        spinchain(SiteType("Fermion"), join(altsystem, altenvironment, sysenvcoupling))

    L = MPO(-im * ad_h, sites)

    cb = SuperfermionCallback(operators, sites, dt)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    state_t = enlargelinks(
        initstate, maxbonddim; ref_state=n -> n ≤ 2nsystem ? "Occ" : "Emp"
    )

    tdvp1vec!(
        state_t,
        L,
        dt,
        tmax,
        sites;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=io_file,
        io_ranks=io_ranks,
        io_times=io_times,
        superfermions=true,
    )

    return nothing
end

function parsecommandline()
    @info "Reading parameters from command line"
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input_parameters", "-i"
        help = "Path to file with JSON dictionary of input parameters"
        arg_type = String
        "--system_sites", "--ns"
        help = "Number of system sites"
        arg_type = Int
        "--environment_sites", "--ne"
        help = "Number of environment sites"
        arg_type = Int
        "--time_step", "--dt"
        help = "Time step of the evolution"
        arg_type = Float64
        "--max_time", "--maxt"
        help = "Total physical time of the evolution"
        arg_type = Float64
        "--bond_dimension", "--bdim"
        help = "Bond dimension of the state MPS"
        arg_type = Int
        "--name", "-o"
        help = "Path of output file"
        arg_type = String
    end

    # Load the input JSON file, if present.
    parsedargs_raw = parse_args(s)
    parsedargs = if haskey(parsedargs_raw, "input_parameters")
        load_pars(pop!(parsedargs_raw, "input_parameters"))
    else
        Dict()
    end

    # Convert keys from String to Symbol and remove the ones whose value is `nothing`.
    for (k, v) in parse_args(s)
        #isnothing(v) || push!(parsedargs, Symbol(k) => v)
        isnothing(v) || push!(parsedargs, k => v)
    end
    return parsedargs
end

function main()
    parsedargs = parsecommandline()

    chain_length = parsedargs["environment_sites"]

    empty_chain_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    empty_chain_coups = CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]
    @show first(empty_chain_coups, 5)

    ops = LocalOperator[]
    for (k, v) in parsedargs["observables"]
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end

    simulation(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=chain_length,
        sysenvcoupling=first(empty_chain_coups),
        environment_chain_frequencies=first(empty_chain_freqs, chain_length),
        environment_chain_couplings=first(empty_chain_coups[2:end], chain_length - 1),
        dt=parsedargs["time_step"],
        tmax=parsedargs["max_time"],
        maxbonddim=parsedargs["bond_dimension"],
        io_file=parsedargs["name"] * "_measurements.csv",
        operators=ops,
    )

    return nothing
end

main()
