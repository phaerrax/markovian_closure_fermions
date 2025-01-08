using ITensors, ITensorMPS, MPSTimeEvolution, ArgParse, CSV, MarkovianClosure
using Base.Iterators: peel

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcoupling,
    nenvironment,
    environment_chain_frequencies,
    environment_chain_couplings,
    nclosure,
    dt,
    tmax,
    maxbonddim,
    io_file=nullfile(),
    io_ranks=nullfile(),
    io_times=nullfile(),
    operators,
)
    system = ModeChain(range(; start=1, step=2, length=nsystem), [system_energy], [])
    altsystem = ModeChain(range(; start=2, step=2, length=nsystem), [system_energy], [])

    environment = ModeChain(
        range(; start=2nsystem + 1, step=2, length=length(environment_chain_frequencies)),
        environment_chain_frequencies,
        environment_chain_couplings,
    )
    altenvironment = ModeChain(
        range(; start=2nsystem + 2, step=2, length=length(environment_chain_frequencies)),
        environment_chain_frequencies,
        environment_chain_couplings,
    )

    truncated_environment, mc = markovianclosure(environment, nclosure, nenvironment)
    truncated_altenvironment, _ = markovianclosure(altenvironment, nclosure, nenvironment)

    closure = ModeChain(
        range(; start=2nsystem + 2nenvironment + 1, step=2, length=nclosure),
        freqs(mc),
        innercoups(mc),
    )
    altclosure = ModeChain(
        range(; start=2nsystem + 2nenvironment + 2, step=2, length=nclosure),
        freqs(mc),
        innercoups(mc),
    )

    sites = siteinds(
        "Fermion", 2nsystem + 2nenvironment + 2nclosure; conserve_nfparity=true
    )
    initstate = MPS(sites, n -> n ≤ 2nsystem ? "Occ" : "Emp")

    ad_h =
        spinchain(
            SiteType("Fermion"), join(system, truncated_environment, sysenvcoupling)
        ) - spinchain(
            SiteType("Fermion"), join(altsystem, truncated_altenvironment, sysenvcoupling)
        ) + spinchain(SiteType("Fermion"), closure) -
        spinchain(SiteType("Fermion"), altclosure)

    # Interaction with last environment site
    for (z, site) in zip(outercoups(mc), closure.range)
        ad_h += z, "cdag", site, "c", last(truncated_environment.range)
        ad_h += conj(z), "cdag", last(truncated_environment.range), "c", site
    end
    for (z, site) in zip(outercoups(mc), altclosure.range)
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironment.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironment.range), "c", site
    end

    # Dissipation operators
    D = OpSum()
    for (g, site, altsite) in zip(damps(mc), closure.range, altclosure.range)
        D += -g, "c", site, "c", altsite
        D += -0.5g, "cdag", site, "c", site
        D += -0.5g, "cdag", altsite, "c", altsite
    end

    L = MPO(-im * ad_h + D, sites)

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

    return state_t
end

function parsecommandline()
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
        "--closure_sites", "--nc"
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

    ops = LocalOperator[]
    for (k, v) in parsedargs["observables"]
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end

    empty_chain_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    sysenvcoupling, empty_chain_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]
    )

    measurements_file = parsedargs["name"] * "_measurements.csv"
    bonddims_file = parsedargs["name"] * "_bonddims.csv"
    simtime_file = parsedargs["name"] * "_simtime.csv"

    @info "You can follow the time evolution step by step in the following files:\n" *
        "$measurements_file\t for the expectation values\n" *
        "$bonddims_file\t for the bond dimensions of the evolved MPS\n" *
        "$simtime_file\t for the wall-clock time spent computing each step"

    simulation(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=parsedargs["environment_sites"],
        nclosure=parsedargs["closure_sites"],
        sysenvcoupling=sysenvcoupling,
        environment_chain_frequencies=empty_chain_freqs,
        environment_chain_couplings=collect(empty_chain_coups),
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
