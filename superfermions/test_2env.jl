using ITensors, ITensorMPS, MPSTimeEvolution, ArgParse, CSV, MarkovianClosure
using Base.Iterators: peel

include("../shared_functions.jl")

function simulation(;
    nsystem,
    system_energy,
    system_initial_state,
    sysenvcouplingL,
    sysenvcouplingR,
    nenvironment,
    environmentL_chain_frequencies,
    environmentL_chain_couplings,
    environmentR_chain_frequencies,
    environmentR_chain_couplings,
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

    # L : initially filled, R : initially empty
    environmentL = ModeChain(
        range(; start=2nsystem + 1, step=4, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )
    altenvironmentL = ModeChain(
        range(; start=2nsystem + 2, step=4, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )

    truncated_environmentL, mcL = markovianclosure(environmentL, nclosure, nenvironment)
    truncated_altenvironmentL, _ = markovianclosure(altenvironmentL, nclosure, nenvironment)

    environmentR = ModeChain(
        range(; start=2nsystem + 3, step=4, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )
    altenvironmentR = ModeChain(
        range(; start=2nsystem + 4, step=4, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )

    truncated_environmentR, mcR = markovianclosure(environmentR, nclosure, nenvironment)
    truncated_altenvironmentR, _ = markovianclosure(altenvironmentR, nclosure, nenvironment)

    environments_last_site = max(
        truncated_environmentL.range[end],
        truncated_altenvironmentL.range[end],
        truncated_environmentR.range[end],
        truncated_altenvironmentR.range[end],
    )

    closureL = ModeChain(
        range(; start=environments_last_site + 1, step=4, length=nclosure),
        freqs(mcL),
        # We hardcode the transformation from empty to filled MC, in which we assume that
        # the alpha coefficients are real. TODO find an automated way to do this.
        -innercoups(mcL),
    )
    altclosureL = ModeChain(
        range(; start=environments_last_site + 2, step=4, length=nclosure),
        freqs(mcL),
        -innercoups(mcL),
    )
    closureR = ModeChain(
        range(; start=environments_last_site + 3, step=4, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )
    altclosureR = ModeChain(
        range(; start=environments_last_site + 4, step=4, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )

    function starts_from_occ(n)
        return (
            in(n, system) ||
            in(n, altsystem) ||
            in(n, truncated_environmentL) ||
            in(n, truncated_altenvironmentL) ||
            in(n, closureL) ||
            in(n, altclosureL)
        )
    end

    site(tags) = addtags(siteind("Fermion"; conserve_nfparity=true), tags)
    sites = [
        site("System")
        site("System(alt)")
        interleave(
            [site("EnvL,n=$n") for n in 1:length(truncated_environmentL)],
            [site("EnvL(alt),n=$n") for n in 1:length(truncated_altenvironmentL)],
            [site("EnvR,n=$n") for n in 1:length(truncated_environmentR)],
            [site("EnvR(alt),n=$n") for n in 1:length(truncated_altenvironmentR)],
        )
        interleave(
            [site("ClosureL,n=$n") for n in 1:length(closureL)],
            [site("ClosureL(alt),n=$n") for n in 1:length(altclosureL)],
            [site("ClosureR,n=$n") for n in 1:length(closureR)],
            [site("ClosureR(alt),n=$n") for n in 1:length(altclosureR)],
        )
    ]

    @assert findall(idx -> hastags(idx, "EnvL"), sites) == truncated_environmentL.range
    @assert findall(idx -> hastags(idx, "EnvR"), sites) == truncated_environmentR.range
    @assert findall(idx -> hastags(idx, "ClosureL"), sites) == closureL.range
    @assert findall(idx -> hastags(idx, "ClosureR"), sites) == closureR.range

    st = SiteType("Fermion")
    initstate = MPS(sites, n -> starts_from_occ(n) ? "Occ" : "Emp")

    # -- Leftover unitary part -------------------------------------------------------------
    ad_h =
        spinchain(st, join(system, truncated_environmentL, sysenvcouplingL)) -
        spinchain(st, join(altsystem, truncated_altenvironmentL, sysenvcouplingR)) +
        spinchain(st, join(system, truncated_environmentR, sysenvcouplingR)) -
        spinchain(st, join(altsystem, truncated_altenvironmentR, sysenvcouplingR))
    # --------------------------------------------------------------------------------------

    # -- Initially filled MC ---------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureL) - spinchain(st, altclosureL)
    # Interaction with last environment site
    for (z, site) in zip(conj.(outercoups(mcL)), closureL.range)
        #                ────    manual empty → filled transformation
        ad_h += z, "cdag", site, "c", last(truncated_environmentL.range)
        ad_h += conj(z), "cdag", last(truncated_environmentL.range), "c", site
    end
    for (z, site) in zip(conj.(outercoups(mcL)), altclosureL.range)
        #                ────    manual empty → filled transformation
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironmentL.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironmentL.range), "c", site
    end
    # Dissipation operator
    DL = OpSum()
    for (g, site, altsite) in zip(damps(mcL), closureL.range, altclosureL.range)
        DL += g, "cdag", site, "cdag", altsite
        DL += -0.5g, "c", site, "cdag", site
        DL += -0.5g, "c", altsite, "cdag", altsite
    end
    # --------------------------------------------------------------------------------------

    # -- Initially empty MC ----------------------------------------------------------------
    # NN interactions
    ad_h += spinchain(st, closureR) - spinchain(st, altclosureR)
    # Interaction with last environment site
    for (z, site) in zip(outercoups(mcR), closureR.range)
        ad_h += z, "cdag", site, "c", last(truncated_environmentR.range)
        ad_h += conj(z), "cdag", last(truncated_environmentR.range), "c", site
    end
    for (z, site) in zip(outercoups(mcR), altclosureR.range)
        ad_h -= z, "cdag", site, "c", last(truncated_altenvironmentR.range)
        ad_h -= conj(z), "cdag", last(truncated_altenvironmentR.range), "c", site
    end
    # Dissipation operator
    DR = OpSum()
    for (g, site, altsite) in zip(damps(mcR), closureR.range, altclosureR.range)
        DR += -g, "c", site, "c", altsite
        DR += -0.5g, "cdag", site, "c", site
        DR += -0.5g, "cdag", altsite, "c", altsite
    end
    # --------------------------------------------------------------------------------------

    L = MPO(-im * ad_h + DL + DR, sites)

    cb = SuperfermionCallback(operators, sites, dt)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    state_t = enlargelinks(
        initstate, maxbonddim; ref_state=n -> starts_from_occ(n) ? "Occ" : "Emp"
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
        "--bond_dimension", "--bdim"
        help = "Bond dimension of the state MPS"
        arg_type = Int
        "--name", "--output", "-o"
        help = "Basename to output files"
        arg_type = String
    end

    # Load the input JSON file, if present.
    @debug "Reading arguments from command line"
    parsedargs_raw = parse_args(s)

    parsedargs = if haskey(parsedargs_raw, "input_parameters")
        input_dict = pop!(parsedargs_raw, "input_parameters")
        @debug "Reading arguments from $input_dict"
        load_pars(input_dict)
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

    ops = LocalOperator[]
    for (k, v) in parsedargs["observables"]
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end

    empty_chainL_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqfilled"]
    sysenvcouplingL, empty_chainL_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupfilled"]
    )
    empty_chainR_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    sysenvcouplingR, empty_chainR_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]
    )

    simulation(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=parsedargs["environment_sites"],
        nclosure=parsedargs["closure_sites"],
        sysenvcouplingL=sysenvcouplingL,
        environmentL_chain_frequencies=empty_chainL_freqs,
        environmentL_chain_couplings=collect(empty_chainL_coups),
        sysenvcouplingR=sysenvcouplingR,
        environmentR_chain_frequencies=empty_chainR_freqs,
        environmentR_chain_couplings=collect(empty_chainR_coups),
        dt=parsedargs["time_step"],
        tmax=parsedargs["max_time"],
        maxbonddim=parsedargs["bond_dimension"],
        io_file=parsedargs["name"] * "_measurements.csv",
        operators=ops,
    )

    return nothing
end

main()
