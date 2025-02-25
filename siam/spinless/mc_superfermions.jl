using OpenSystemsChainMapping
using ITensorMPS: maxlinkdim, siteinds
using MPSTimeEvolution: SuperfermionCallback, tdvp1vec!
using Base.Iterators: peel
using HDF5, CSV

function main()
    parsedargs = parsecommandline()

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

    initstate, L = siam_spinless_superfermions_mc(;
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
            @error "Increasing bond dimension of a state input from file not implemented."
        end
    end

    dt = parsedargs["time_step"]
    tmax = parsedargs["max_time"]
    operators = parseoperators(parsedargs["observables"])
    cb = SuperfermionCallback(operators, siteinds(initstate), dt)

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
        superfermions=true,
    )

    pack!(
        parsedargs["output"] * ".h5";
        argsdict=parsedargs,
        expvals_file=measurements_file,
        bonddimensions_file=bonddims_file,
        walltime_file=simtime_file,
    )

    return nothing
end

main()
