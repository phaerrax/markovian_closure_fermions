using ITensors, ITensorMPS, MPSTimeEvolution, ArgParse, CSV

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

    simulation_files_info(;
        measurements_file=io_file, bonddims_file=io_ranks, simtime_file=io_times
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
        "--max_bond_dimension", "--bdim"
        help = "Bond dimension of the state MPS"
        arg_type = Int
        "--name", "--output", "-o"
        help = "Basename to output files"
        arg_type = String
    end

    # Load the input JSON file, if present.
    parsedargs_raw = parse_args(s)
    parsedargs = if haskey(parsedargs_raw, "input_parameters")
        inputfile = pop!(parsedargs_raw, "input_parameters")
        @info "Reading parameters from $inputfile and from the command line"
        load_pars(inputfile)
    else
        @info "Reading parameters from the command line"
        Dict()
    end

    # Convert keys from String to Symbol and remove the ones whose value is `nothing`.
    for (k, v) in parse_args(s)
        isnothing(v) || push!(parsedargs, k => v)
    end
    return parsedargs
end

function main()
    parsedargs = parsecommandline()

    chain_length = parsedargs["environment_sites"]

    empty_chain_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    empty_chain_coups = CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]

    ops = LocalOperator[]
    for (k, v) in parsedargs["observables"]
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end

    measurements_file = parsedargs["name"] * "_measurements.csv"
    bonddims_file = parsedargs["name"] * "_bonddims.csv"
    simtime_file = parsedargs["name"] * "_simtime.csv"

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
        maxbonddim=parsedargs["max_bond_dimension"],
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
        operators=ops,
    )

    pack!(
        parsedargs["name"] * ".h5";
        argsdict=parsedargs,
        expvals_file=measurements_file,
        bonddimensions_file=bonddims_file,
        walltime_file=simtime_file,
    )

    return nothing
end

main()
