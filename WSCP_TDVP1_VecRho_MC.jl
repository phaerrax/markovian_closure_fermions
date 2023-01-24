using ITensors
using DelimitedFiles
using Pkg
using PseudomodesTTEDOPA

#Pkg.activate(".")
#using TimeEvoVecMPS
#Occhio qui. Va sistemato in modo che tutto
#sia portabile su diverse macchine
#REMINDER: INDACO

#modificato prima di andare via 06/10/22
include("./TDVP_lib_VecRho.jl")

let
    # Load parameters from JSON
    parameters = load_pars(ARGS[1])

    # Select TEDOPA/+MC
    isMC = parameters["withMC"]

    # Define overall system (sys+chain) and initial state
    sysenv, psi0 = defineSystem(;
        sys_type="HvS=1/2",
        sys_istate=parameters["sys_ini"],
        chain_size=parameters["chain_length"],
        local_dim=parameters["chain_loc_dim"],
    )

    if isMC
        println("TEDOPA+MC")
        H = createMPOVecRho(
            sysenv,
            parameters["sys_en"],
            parameters["sys_coup"],
            parameters["chain_freqs"],
            parameters["chain_coups"],
            parameters["MC_alphas"],
            parameters["MC_betas"],
            parameters["MC_coups"],
            parameters["omegaInf"],
        )
    else
        # Create an Hamiltonian specialized on sysenv MPS form
        println("Standard TEDOPA")
        H = createMPO(
            sysenv,
            parameters["sys_en"],
            parameters["sys_coup"],
            parameters["chain_freqs"],
            parameters["chain_coups"],
        )
    end

    # Define quantities that must be observed
    obpairs = [["Norm", 1], ["vecσx", 1], ["vecσz", 1]]
    for i in 10:10:80
        push!(obpairs, ["vecN", i + 1])
    end
    for i in 81:1:86
        push!(obpairs, ["vecN", i + 1])
    end
    vobs = createObs(obpairs)

    # Debug

    # Copy initial state into evolving state.
    psi, overlap = stretchBondDim(psi0, 20)

    #First measurement (t=0)

    # appo = Float64[]
    # for lookat in vobs
    #    orthogonalize!(psi,lookat.pos); 
    #    psidag = dag(prime(psi[lookat.pos],"Site"));
    #    m=scalar(psidag*op(sysenv,lookat.op,lookat.pos)*psi[lookat.pos]);
    #    push!(appo,m);  
    # end

    # println(appo);

    # flush(stdout);

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    cbT = LocalPosMeasurementCallback(vobs, sysenv, parameters["ms_stride"] * timestep)

    if isMC
        # Disable "progress" to run on the cluster
        tdvp1!(
            psi,
            H,
            timestep,
            tmax;
            hermitian=false,
            normalize=false,
            callback=cbT,
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            store_psi0=true,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    else
        tdvp1!(
            psi,
            H,
            timestep,
            tmax;
            hermitian=true,
            normalize=true,
            callback=cbT,
            progress=true,
            exp_tol=parameters["exp_tol"],
            krylovdim=parameters["krylov_dim"],
            store_psi0=true,
            io_file=parameters["out_file"],
            io_ranks=parameters["ranks_file"],
            io_times=parameters["times_file"],
        )
    end
end
