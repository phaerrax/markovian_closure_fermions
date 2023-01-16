using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA

#using TimeEvoVecMPS
#Occhio qui. Va sistemato in modo che tutto
#sia portabile su diverse macchine
#REMINDER: INDACO

#modificato prima di andare via 06/10/22
include("./TDVP_lib_VecRho.jl")

#Load parameters from JSON
parameters = load_parameters("provaMC.json")

#Select TEDOPA/+MC
isMC = parameters["withMC"]

#Define overall system (sys+chain) and initial state``
sysenv, psi0 = defineSystem(;
    sys_type   = "HvS=1/2",
    sys_istate = parameters["sys_ini"],
    chain_size = parameters["chain_length"],
    local_dim  = parameters["chain_loc_dim"],
)

if !isMC
    # Create Hamiltonian specialized on sysenv MPS form
    println("Standard TEDOPA")
    H = createMPO(
        sysenv,
        parameters["sys_en"],
        parameters["sys_coup"],
        parameters["chain_freqs"],
        parameters["chain_coups"],
    )
else
    println("TEDOPA+MC")
    H = createMPOVecRho(
        sysenv,
        parameters["sys_en"],      # Energy of the system site
        parameters["sys_coup"],    # System-environment coupling
        parameters["chain_freqs"], # Frequencies of chain sites
        parameters["chain_coups"], # Couplings between chain sites
        parameters["MC_alphas"],   # ???
        parameters["MC_betas"],    # ???
        parameters["MC_coups"],    # Couplings between pseudomodes?
        parameters["omegaInf"],    # ???
    )
end
                  
# Define quantities that must be observed.
# Firstly, the nom, σˣ and σᶻ on the system site.
obs = [["Norm", 1], ["vecσx", 1], ["vecσz", 1]]
# Then, the occupation number in some sites of the chain...
for i in 10:10:80
    push!(obs, ["vecN", i + 1])
end
# ...and in each pseudomode.
for i in 81:86
    push!(obs, ["vecN", i + 1])
end
vobs = createObs(obs)
# `vobs` isa Vector{opPos}: each element is a couple of elements, the first is an operator
# and the second a position (aka a site along the chain).

# Enlarge the bond dimensions so that TDVP1 has the possibility to grow
# the number of singular values between the bonds.
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

cbT = LocalMeasurementCallbackTama(vobs, sysenv, parameters["ms_stride"] * timestep)
# Create a LocalMeasurementCallback where we append to the names of each operator a
# subscript with the site to which it refers.

if !isMC
    # If we are studying the Markovian closure, the evolution is not unitary, so we
    # should not indicate that the "Hamiltonian" (which is actually the Lindbladian
    # superoperator) is Hermitian, nor should we normalize the state (the vectorized
    # density matrix does not have unit norm, generally).
    #
    # Disable "progress" to run on the cluster.
    tdvp1!(
        psi,
        H,
        timestep,
        tmax;
        hermitian  = true,
        normalize  = true,
        callback   = cbT,
        progress   = false,
        exp_tol    = parameters["exp_tol"],
        krylovdim  = parameters["krylov_dim"],
        store_psi0 = true,
        io_file    = parameters["out_file"],
        io_ranks   = parameters["ranks_file"],
        io_times   = parameters["times_file"],
    )
else
    tdvp1!(
        psi,
        H,
        timestep,
        tmax;
        hermitian  = false,
        normalize  = false,
        callback   = cbT,
        progress   = false,
        exp_tol    = parameters["exp_tol"],
        krylovdim  = parameters["krylov_dim"],
        store_psi0 = true,
        io_file    = parameters["out_file"],
        io_ranks   = parameters["ranks_file"],
        io_times   = parameters["times_file"],
    )
end
