using OpenSystemsChainMapping
using ITensorMPS: maxlinkdim, siteinds
using MPSTimeEvolution: ExpValueCallback, tdvp1!, adaptivetdvp1!
using Base.Iterators: peel
using HDF5, CSV

function main()
    parsedargs = parsecommandline(
        ["--save_final_state"],
        Dict(:help => "store final state in the output file", :action => :store_true),
        ["--adaptive_tdvp_convergence_factor"],
        Dict(:help => "convergence factor for adaptive TDVP1", :arg_type => Float64),
    )

    cfs = if haskey(parsedargs, "environment_chain_coefficients")
        cf_file = CSV.File(parsedargs["environment_chain_coefficients"])
        Dict(
            :empty =>
                (frequencies=cf_file["freqempty"], couplings=cf_file["coupempty"]),
            :filled =>
                (frequencies=cf_file["freqfilled"], couplings=cf_file["coupfilled"]),
        )
    else
        tedopa_chain_coefficients(; parsedargs...)
    end

    empty_chainL_freqs = cfs[:filled].frequencies
    sysenvcouplingL, empty_chainL_coups = peel(cfs[:filled].couplings)
    empty_chainR_freqs = cfs[:empty].frequencies
    sysenvcouplingR, empty_chainR_coups = peel(cfs[:empty].couplings)

    set_bond_dimension = parsedargs["max_bond_dimension"]
    measurements_file = parsedargs["output"] * "_measurements.csv"
    bonddims_file = parsedargs["output"] * "_bonddims.csv"
    simtime_file = parsedargs["output"] * "_simtime.csv"

    initstate, H = siam_spinless_pure_state(;
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
        maxbonddim=if haskey(parsedargs, "adaptive_tdvp_convergence_factor")
            2
        else
            set_bond_dimension
        end,
        conserve_nf=(!haskey(parsedargs, "adaptive_tdvp_convergence_factor")),
        conserve_nfparity=(!haskey(parsedargs, "adaptive_tdvp_convergence_factor")),
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
    cb = ExpValueCallback(operators, siteinds(initstate), dt)

    tdvp1_kwargs = (
        callback=cb,
        hermitian=true,
        io_file=measurements_file,
        io_ranks=bonddims_file,
        io_times=simtime_file,
    )

    if haskey(parsedargs, "adaptive_tdvp_convergence_factor")
        f = parsedargs["adaptive_tdvp_convergence_factor"]
        @info "Running adaptive TDVP1 algorithm with tolerance $f"
        adaptivetdvp1!(
            initstate, H, dt, tmax; convergence_factor_bonddims=f, tdvp1_kwargs...
        )
    else
        @info "Running standard TDVP1 algorithm"
        tdvp1!(initstate, H, dt, tmax; tdvp1_kwargs...)
    end

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
