using ITensors, ITensorMPS, LindbladVectorizedTensors, MarkovianClosure, MPSTimeEvolution
using HDF5, CSV
using Base.Iterators: peel

include("../../shared_functions.jl")

function ITensors.state(sn::StateName"vF", st::SiteType"vFermion")
    return LindbladVectorizedTensors.vop(sn, st)
end

function siam_spinless_2env_mc(;
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
    maxbonddim,
)
    system = ModeChain(range(; start=1, step=1, length=nsystem), [system_energy], [])

    # L : initially filled, R : initially empty
    environmentL = ModeChain(
        range(; start=nsystem + 1, step=2, length=length(environmentL_chain_frequencies)),
        environmentL_chain_frequencies,
        environmentL_chain_couplings,
    )
    environmentR = ModeChain(
        range(; start=nsystem + 2, step=2, length=length(environmentR_chain_frequencies)),
        environmentR_chain_frequencies,
        environmentR_chain_couplings,
    )

    truncated_environmentL, mcL = markovianclosure(environmentL, nclosure, nenvironment)
    truncated_environmentR, mcR = markovianclosure(environmentR, nclosure, nenvironment)

    environments_last_site = maximum(
        [truncated_environmentL.range; truncated_environmentR.range]
    )

    closureL = ModeChain(
        range(; start=environments_last_site + 1, step=2, length=nclosure),
        freqs(mcL),
        # We hardcode the transformation from empty to filled MC, in which we assume that
        # the alpha coefficients are real. TODO find an automated way to do this.
        -innercoups(mcL),
    )
    closureR = ModeChain(
        range(; start=environments_last_site + 2, step=2, length=nclosure),
        freqs(mcR),
        innercoups(mcR),
    )

    function init(n)
        return if in(n, system)
            system_initial_state
        elseif in(n, truncated_environmentL) || in(n, closureL)
            "Occ"
        else
            "Emp"
        end
    end

    st = SiteType("vFermion")
    site(tags) = addtags(siteind("vFermion"), tags)
    sites = [
        site("System")
        interleave(
            [site("EnvL") for n in 1:nenvironment], [site("EnvR") for n in 1:nenvironment]
        )
        interleave(
            [site("ClosureL") for n in 1:nclosure], [site("ClosureR") for n in 1:nclosure]
        )
    ]
    for n in eachindex(sites)
        sites[n] = addtags(sites[n], "n=$n")
    end
    initstate = MPS(sites, init)

    @assert findall(idx -> hastags(idx, "System"), sites) == system.range
    @assert findall(idx -> hastags(idx, "EnvL"), sites) == truncated_environmentL.range
    @assert findall(idx -> hastags(idx, "EnvR"), sites) == truncated_environmentR.range
    @assert findall(idx -> hastags(idx, "ClosureL"), sites) == closureL.range
    @assert findall(idx -> hastags(idx, "ClosureR"), sites) == closureR.range

    ad_h = OpSum()
    DL = OpSum()
    DR = OpSum()

    ad_h = spinchain(
        SiteType("vFermion"),
        join(
            reverse(truncated_environmentR),
            join(system, truncated_environmentL, sysenvcouplingL),
            sysenvcouplingR,
        ),
    )

    # -- Initially filled MC ---------------------------------------------------------------
    # NN interactions
    ad_h_mcL = spinchain(st, closureL)
    # Interaction with last environment site
    chain_edge_site = last(truncated_environmentL.range)
    for (n, z) in zip(closureL.range, outercoups(mcL))
        jws = jwstring(; start=chain_edge_site, stop=n)
        ad_h_mcL += (
            z * gkslcommutator("A†", chain_edge_site, jws..., "A", n) +
            conj(z) * gkslcommutator("A", chain_edge_site, jws..., "A†", n)
        )
    end
    # Dissipation operator
    for (n, g) in zip(closureL.range, damps(mcL))
        # a† ρ a
        opstring = [repeat(["F⋅ * ⋅F"], n - 1); "A†⋅ * ⋅A"]
        DL += (g, interleave(opstring, 1:n)...)
        # -0.5 (a a† ρ + ρ a a†) = 0.5 (a† a ρ + ρ a† a) - ρ
        DL += 0.5g, "N⋅", n
        DL += 0.5g, "⋅N", n
        DL += -g, "Id", n
    end
    # --------------------------------------------------------------------------------------

    # -- Initially empty MC ----------------------------------------------------------------
    # NN interactions
    ad_h_mcR = spinchain(st, closureR)
    # Interaction with last environment site
    chain_edge_site = last(truncated_environmentR.range)
    for (n, z) in zip(closureR.range, outercoups(mcR))
        jws = jwstring(; start=chain_edge_site, stop=n)
        ad_h_mcR += (
            conj(z) * gkslcommutator("A†", chain_edge_site, jws..., "A", n) +
            z * gkslcommutator("A", chain_edge_site, jws..., "A†", n)
        )
    end
    # Dissipation operator
    for (n, g) in zip(closureR.range, damps(mcR))
        # a ρ a†
        opstring = [repeat(["F⋅ * ⋅F"], n - 1); "A⋅ * ⋅A†"]
        DR += (g, interleave(opstring, 1:n)...)
        # -0.5 (a† a ρ + ρ a† a)
        DR += -0.5g, "N⋅", n
        DR += -0.5g, "⋅N", n
    end
    # --------------------------------------------------------------------------------------

    @show ad_h
    @show ad_h_mcL
    @show ad_h_mcR
    @show DL
    @show DR
    L = MPO(ad_h + ad_h_mcL + ad_h_mcR + DL + DR, sites)

    # Prepare the state for TDVP with long-range interactions by artificially increasing
    # its bond dimensions.
    growMPS!(initstate, maxbonddim)

    return initstate, L
end

function main()
    parsedargs = parsecommandline(
        ["--save_final_state"],
        Dict(:help => "store final state in the output file", :action => :store_true),
    )

    empty_chainL_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqfilled"]
    sysenvcouplingL, empty_chainL_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupfilled"]
    )
    empty_chainR_freqs = CSV.File(parsedargs["environment_chain_coefficients"])["freqempty"]
    sysenvcouplingR, empty_chainR_coups = peel(
        CSV.File(parsedargs["environment_chain_coefficients"])["coupempty"]
    )

    set_bond_dimension = parsedargs["max_bond_dimension"]
    measurements_file = parsedargs["output"] * "_measurements.csv"
    bonddims_file = parsedargs["output"] * "_bonddims.csv"
    simtime_file = parsedargs["output"] * "_simtime.csv"

    initstate, L = siam_spinless_2env_mc(;
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
        maxbonddim=set_bond_dimension,
    )

    if haskey(parsedargs, "initial_state_file")
        initstate_file = parsedargs["initial_state_file"]
        # Discard prepared state and load from file
        initstate = h5open(initstate_file, "r") do file
            return read(file, parsedargs["initial_state_label"], MPS)
        end
        # Increase bond dimension if needed
        if maxlinkdim(initstate) < set_bond_dimension
            initstate = enlargelinks(initstate, set_bond_dimension)
        end
    end

    dt = parsedargs["time_step"]
    tmax = parsedargs["max_time"]
    operators = parseoperators(parsedargs["observables"])
    cb = ExpValueCallback(operators, siteinds(initstate), dt)

    simulation_files_info(;
        measurements_file=measurements_file,
        bonddims_file=bonddims_file,
        simtime_file=simtime_file,
    )

    tdvp1vec!(
        initstate,
        L,
        dt,
        tmax;
        callback=cb,
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
    )

    pack!(
        parsedargs["output"] * ".h5";
        argsdict=parsedargs,
        expvals_file=measurements_file,
        bonddimensions_file=bonddims_file,
        walltime_file=simtime_file,
        finalstate=parsedargs["save_final_state"] ? initstate : nothing,
    )

    return nothing
end

main()
