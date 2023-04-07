#!/usr/bin/julia

using ITensors, DelimitedFiles, Printf
using ITensorTDVP, Observers, PseudomodesTTEDOPA

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO e il TDVP in-house di ITensor.

include("TDVP_lib_VecRho.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]

    # Input: TTEDOPA chain parameters
    tedopa_coefficients = readdlm(
        parameters["tedopa_coefficients"], ',', Float64; skipstart=1
    )
    coups = tedopa_coefficients[:, 1]
    freqs = tedopa_coefficients[:, 2]
    chain_length = parameters["chain_length"]

    sites = siteinds("S=1/2", system_length + chain_length)
    psi0 = MPS(sites, [system_initstate; repeat(["Dn"], chain_length)])
    psi, _ = stretchBondDim(psi0, parameters["max_bond"])

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]
    # Hamiltonian of the system:
    h = OpSum()

    # - system Hamiltonian
    h += eps, "N", 1

    # - system-chain interaction
    if lowercase(parameters["interaction_type"]) == "xx"
        # The XX interaction with the chain, with the Jordan-Wigner transformation, is
        #   i(c0 + c0†)(c1 + c1†) = - σy ⊗ σx.
        h += -4 * coups[1], "Sy", 1, "Sx", 2
    elseif lowercase(parameters["interaction_type"]) == "exchange"
        h += coups[1], "S+", 1, "S-", 2
        h += coups[1], "S-", 1, "S+", 2
    else
        throw(error("Unrecognized interaction type. Please use \"xx\" or \"exchange\"."))
    end

    # - TTEDOPA chain
    for j in system_length .+ (1:chain_length)
        h += freqs[j - 1], "N", j
    end

    for j in system_length .+ (1:(chain_length - 1))
        h += coups[j], "S-", j, "S+", j + 1
        h += coups[j], "S+", j, "S-", j + 1
    end

    H = MPO(h, sites)
    n1 = op(sites[1], "N")

    function sysn(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return real(scalar(dag(psi[1]) * noprime(n1 * psi[1])))
        end
        return nothing
    end
    function currenttime(; current_time, bond, half_sweep)
        # Get the times at which the observable are computed.
        if bond == 1 && half_sweep == 2
            return -imag(current_time)
            # The TDVP is run with imaginary time steps (look below).
        end
        return nothing
    end
    function normalization(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return norm(psi)
        end
        return nothing
    end

    obs = Observer("norm" => normalization, "time" => currenttime, "sysn" => sysn)

    stime = @elapsed begin
        ψf = tdvp(
            #tdvp_solver(; solver_backend="exponentiate"),
            # exponentiate_solver(
            # exp_tol=parameters["exp_tol"],
            # krylovdim=parameters["krylov_dim"],
            #),
            H,
            -im * tmax,
            psi;
            time_step=-im * timestep,
            normalize=true,
            (observer!)=obs,
            mindim=parameters["max_bond"],
            maxdim=parameters["max_bond"]+1
            #cutoff=parameters["discarded_w"],
        )
        #mindim=parameters["MP_minimum_bond_dimension"],
        #maxdim=parameters["MP_maximum_bond_dimension"],

        # A partire dai risultati costruisco delle matrici da dare poi in pasto
        # alle funzioni per i grafici e le tabelle di output
        io_handle = open(parameters["out_file"], "w")
        @printf(io_handle, "%20s", "time")
        @printf(io_handle, "%20s", "norm")
        @printf(io_handle, "%20s", "sysn")
        @printf(io_handle, "\n")
        tout = results(obs, "time")
        statenorm = results(obs, "norm")
        occnlist = results(obs, "sysn")
        for row in zip(tout, statenorm, occnlist)
            for datum in row
                @printf(io_handle, "%20.15f", datum)
            end
            @printf(io_handle, "\n")
            flush(io_handle)
        end
        close(io_handle)
    end
    println("Elapsed time: $stime")

    return nothing
end
