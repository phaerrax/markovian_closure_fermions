using ITensors, ITensorMPS, LindbladVectorizedTensors, MarkovianClosure, MPSTimeEvolution
using HDF5, CSV
using Base.Iterators: peel

include("../../shared_functions.jl")

function siam_spinless_tedopa(;
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
    maxbonddim,
)
    system = ModeChain(1:nsystem, [system_energy], [])
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
    environmentL = first(environmentL, nenvironment)
    environmentR = first(environmentR, nenvironment)

    function init(n)
        return if in(n, system)
            system_initial_state
        elseif in(n, environmentL)
            "Occ"
        else
            "Emp"
        end
    end
    st = SiteType("Fermion")
    site(tags) = addtags(siteind("Fermion"), tags)
    sites = [
        site("System")
        interleave(
            [site("EnvL") for n in 1:nenvironment], [site("EnvR") for n in 1:nenvironment]
        )
    ]
    for n in eachindex(sites)
        sites[n] = addtags(sites[n], "n=$n")
    end
    initstate = MPS(sites, init)

    @assert findall(idx -> hastags(idx, "System"), sites) == system.range
    @assert findall(idx -> hastags(idx, "EnvL"), sites) == environmentL.range

    h = spinchain(
        SiteType("Fermion"),
        join(
            reverse(environmentR),
            join(system, environmentL, sysenvcouplingL),
            sysenvcouplingR,
        ),
    )
    H = MPO(h, sites)

    initstate = enlargelinks(initstate, maxbonddim; ref_state=init)

    return initstate, H
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

    initstate, H = siam_spinless_tedopa(;
        nsystem=parsedargs["system_sites"],
        system_energy=parsedargs["system_energy"],
        system_initial_state=parsedargs["system_initial_state"],
        nenvironment=parsedargs["environment_sites"],
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

    tdvp1!(
        initstate,
        H,
        dt,
        tmax;
        callback=cb,
        hermitian=true,
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
