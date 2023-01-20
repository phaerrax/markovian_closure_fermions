using ITensors
using JSON
using DelimitedFiles
using Permutations
using TimeEvoVecMPS
using PseudomodesTTEDOPA

"""
    load_pars(file_name::String)

Load the JSON file `file_name` into a dictionary.
"""
function load_pars(file_name::String)
    input = open(file_name)
    s = read(input, String)
    # Aggiungo anche il nome del file alla lista di parametri.
    p = JSON.parse(s)
    return p
end

function defineSystem(;
    sys_type::String="HvS=1/2",
    sys_istate::String="Up",
    chain_size::Int64=250,
    local_dim::Int64=6,
)
    sys = siteinds(sys_type, 1)
    env = siteinds("HvOsc", chain_size; dim=local_dim)
    sysenv = vcat(sys, env)

    stateSys = [sys_istate]

    #Standard approach: chain always in the vacuum state
    stateEnv = ["0" for n in 1:chain_size]

    stateSE = vcat(stateSys, stateEnv)

    psi0 = productMPS(sysenv, stateSE)

    println("Initial MPS_vecRho state of sys-env set")

    return (sysenv, psi0)
end

#function setInital()
function createMPO(
    sysenv, eps::Float64, delta::Float64, freqfile::String, coupfile::String
)::MPO
    coups = readdlm(coupfile)
    freqs = readdlm(freqfile)

    NN::Int64 = size(sysenv)[1]
    NChain::Int64 = NN - 1

    thempo = OpSum()
    #system Hamiltonian
    #pay attention to constant S_x/y/z = 0.5 σ_x/y/z
    thempo += 2 * eps, "Sz", 1
    thempo += 2 * delta, "Sx", 1
    #system-env interaction
    #ATTENTION:
    #!Sx = 0.5 σx
    thempo += 2 * coups[1], "Sx", 1, "Adag", 2
    thempo += 2 * coups[1], "Sx", 1, "A", 2
    #chain local Hamiltonians
    for j in 2:NChain
        thempo += freqs[j - 1], "N", j
    end

    for j in 2:(NChain - 1)
        thempo += coups[j], "A", j, "Adag", j + 1
        thempo += coups[j], "Adag", j, "A", j + 1
    end
    return MPO(thempo, sysenv)
end

#For MC
function createMPO(
    sysenv,
    eps::Float64,
    delta::Float64,
    freqfile::String,
    coupfile::String,
    MC_alphafile::String,
    MC_betafile::String,
    MC_coupfile::String,
    omega::Float64;
    kwargs...,
)::MPO

    #perm:  permutation of the closure oscillators
    perm = get(kwargs, :perm, nothing)

    #Chain parameters
    coups = readdlm(coupfile)
    freqs = readdlm(freqfile)

    #Closure parameters loaded from file
    alphas_MC = readdlm(MC_alphafile)
    betas_MC = readdlm(MC_betafile)
    coups_MC = readdlm(MC_coupfile)
    gammas = omega * alphas_MC[:, 1]
    eff_freqs = [omega + 1im * g for g in gammas]
    eff_gs = omega * betas_MC[:, 2]
    eff_coups = omega / 2 * (coups_MC[:, 1] + 1im * coups_MC[:, 2])

    NN = length(sysenv)
    MC_N = length(gammas)
    NP_Chain = NN - MC_N

    if (perm != nothing)
        if (length(perm) != MC_N)
            println("The provided permutation is not correct")
        end

        pmtx = Permutation(perm)
        @show pmtx
    else
        #Identity permutation
        pmtx = Permutation(collect(1:MC_N))
        @show pmtx
    end

    #Lavoriamo qui
    thempo = OpSum()
    #system Hamiltonian
    #pay attention to constant S_x/y/z = 0.5 σ_x/y/z
    thempo += 2 * eps, "Sz", 1
    thempo += 2 * delta, "Sx", 1
    #system-env interaction
    #!Sx = 0.5 σx
    thempo += 2 * coups[1], "Sx", 1, "Adag", 2
    thempo += 2 * coups[1], "Sx", 1, "A", 2

    #Primary chain local Hamiltonians
    for j in 2:NP_Chain
        thempo += freqs[j - 1], "N", j
    end

    for j in 2:(NP_Chain - 1)
        thempo += coups[j], "A", j, "Adag", j + 1
        thempo += coups[j], "Adag", j, "A", j + 1
    end
    #################################

    #Markovian closure Hamiltonian
    for j in 1:MC_N
        thempo += eff_freqs[j], "N", NP_Chain + pmtx(j)
    end

    for j in 1:(MC_N - 1)
        thempo += eff_gs[j], "A", NP_Chain + pmtx(j), "Adag", NP_Chain + pmtx(j + 1)
        thempo += eff_gs[j], "Adag", NP_Chain + pmtx(j), "A", NP_Chain + pmtx(j + 1)
    end

    #################################

    #Primary chain - MC interaction
    for j in 1:MC_N
        thempo += eff_coups[j], "A", NP_Chain, "Adag", NP_Chain + pmtx(j)
        thempo += conj(eff_coups[j]), "Adag", NP_Chain, "A", NP_Chain + pmtx(j)
    end
    #################################

    return MPO(thempo, sysenv)
end
function createMPO2MC(
    sysenv,
    eps::Float64,
    delta::Float64,
    freqfile::String,
    coupfile::String,
    MC_alphafile::String,
    MC_betafile::String,
    MC_coupfile::String,
    omega::Float64;
    kwargs...,
)::MPO

    #perm:  permutation of the closure oscillators
    perm = get(kwargs, :perm, nothing)

    #Chain parameters
    coups = readdlm(coupfile)
    freqs = readdlm(freqfile)

    #Closure parameters loaded from file
    alphas_MC = readdlm(MC_alphafile)
    betas_MC = readdlm(MC_betafile)
    coups_MC = readdlm(MC_coupfile)
    gammas = omega * alphas_MC[:, 1]
    eff_freqs = [omega + 1im * g for g in gammas]
    eff_gs = omega * betas_MC[:, 2]
    #Reduce interaction with each closure
    eff_coups = 1 / sqrt(2) * omega / 2 * (coups_MC[:, 1] + 1im * coups_MC[:, 2])

    NN = length(sysenv)
    MC_N = length(gammas)
    NP_Chain = NN - 2 * MC_N

    if (perm != nothing)
        if (length(perm) != MC_N)
            println("The provided permutation is not correct")
        end

        pmtx = Permutation(perm)
        @show pmtx
    else
        #Identity permutation
        pmtx = Permutation(collect(1:MC_N))
        @show pmtx
    end

    #Lavoriamo qui
    thempo = OpSum()
    #system Hamiltonian
    #pay attention to constant S_x/y/z = 0.5 σ_x/y/z
    thempo += 2 * eps, "Sz", 1
    thempo += 2 * delta, "Sx", 1
    #system-env interaction
    #!Sx = 0.5 σx
    thempo += 2 * coups[1], "Sx", 1, "Adag", 2
    thempo += 2 * coups[1], "Sx", 1, "A", 2

    #Primary chain local Hamiltonians
    for j in 2:NP_Chain
        thempo += freqs[j - 1], "N", j
    end

    for j in 2:(NP_Chain - 1)
        thempo += coups[j], "A", j, "Adag", j + 1
        thempo += coups[j], "Adag", j, "A", j + 1
    end
    #################################

    #First+Second Markovian closure Hamiltonian
    for j in 1:MC_N
        thempo += eff_freqs[j], "N", NP_Chain + pmtx(j)
        #Second closure
        thempo += eff_freqs[j], "N", NP_Chain + MC_N + pmtx(j)
    end

    for j in 1:(MC_N - 1)
        thempo += eff_gs[j], "A", NP_Chain + pmtx(j), "Adag", NP_Chain + pmtx(j + 1)
        thempo += eff_gs[j], "Adag", NP_Chain + pmtx(j), "A", NP_Chain + pmtx(j + 1)
        #Second clousure
        thempo += eff_gs[j],
        "A", NP_Chain + MC_N + pmtx(j), "Adag",
        NP_Chain + MC_N + pmtx(j + 1)
        thempo += eff_gs[j],
        "Adag", NP_Chain + MC_N + pmtx(j), "A",
        NP_Chain + MC_N + pmtx(j + 1)
    end

    #################################

    #Primary chain - MC interaction
    for j in 1:MC_N
        thempo += eff_coups[j], "A", NP_Chain, "Adag", NP_Chain + pmtx(j)
        thempo += conj(eff_coups[j]), "Adag", NP_Chain, "A", NP_Chain + pmtx(j)
        #Second closure
        thempo += eff_coups[j], "A", NP_Chain, "Adag", NP_Chain + MC_N + pmtx(j)
        thempo += conj(eff_coups[j]), "Adag", NP_Chain, "A", NP_Chain + MC_N + pmtx(j)
    end
    #################################

    return MPO(thempo, sysenv)
end

function createMPOVecRho(
    sysenv,
    eps::Float64,
    delta::Float64,
    freqfile::String,
    coupfile::String,
    MC_alphafile::String,
    MC_betafile::String,
    MC_coupfile::String,
    omega::Float64;
    kwargs...,
)

    #perm:  permutation of the closure oscillators
    perm = get(kwargs, :perm, nothing)

    #Chain parameters
    coups = readdlm(coupfile)
    freqs = readdlm(freqfile)

    #Closure parameters loaded from file
    alphas_MC = readdlm(MC_alphafile)
    betas_MC = readdlm(MC_betafile)
    coups_MC = readdlm(MC_coupfile)
    gammas = omega * alphas_MC[:, 1]
    eff_freqs = [omega + 0.0 for g in gammas]
    eff_gs = omega * betas_MC[:, 2]
    eff_coups = omega / 2 * (coups_MC[:, 1] + 1im * coups_MC[:, 2])

    NN = length(sysenv)
    MC_N = length(gammas)
    NP_Chain = NN - MC_N

    if (perm != nothing)
        if (length(perm) != MC_N)
            println("The provided permutation is not correct")
        end

        pmtx = Permutation(perm)
        @show pmtx
    else
        #Identity permutation
        pmtx = Permutation(collect(1:MC_N))
        @show pmtx
    end

    #Lavoriamo qui
    thempo = OpSum()
    #system Hamiltonian
    #pay attention to constants vecσ_x/z/v =  σ_x/y/z

    #We start by the von Neumann part
    #-i[H,ρ] = -i H ρ + i ρ H
    #Left action
    thempo += -1im * eps, "σz⋅", 1
    thempo += -1im * delta, "σx⋅", 1

    #Right action
    thempo += +1im * eps, "⋅σz", 1
    thempo += +1im * delta, "⋅σx", 1

    #system-env interaction
    #!vecσx = σx
    #Debug
    print(coups[1])
    #Debug
    # coups[1]=0.
    #Left action
    thempo += -1im * coups[1], "σx⋅", 1, "a+⋅", 2
    thempo += -1im * coups[1], "σx⋅", 1, "a-⋅", 2
    #Right action
    thempo += +1im * coups[1], "⋅σx", 1, "⋅a+", 2
    thempo += +1im * coups[1], "⋅σx", 1, "⋅a-", 2

    #Primary chain local Hamiltonians
    for j in 2:NP_Chain
        thempo += -1im * freqs[j - 1], "N⋅", j
        thempo += +1im * freqs[j - 1], "⋅N", j
    end

    for j in 2:(NP_Chain - 1)
        thempo += -1im * coups[j], "a-⋅", j, "a+⋅", j + 1
        thempo += -1im * coups[j], "a+⋅", j, "a-⋅", j + 1

        thempo += +1im * coups[j], "⋅a-", j, "⋅a+", j + 1
        thempo += +1im * coups[j], "⋅a+", j, "⋅a-", j + 1
    end
    #################################

    #Markovian closure Hamiltonian
    for j in 1:MC_N
        thempo += -1im * eff_freqs[j], "N⋅", NP_Chain + pmtx(j)
        thempo += +1im * eff_freqs[j], "⋅N", NP_Chain + pmtx(j)
    end

    for j in 1:(MC_N - 1)
        thempo += -1im * eff_gs[j], "a-⋅", NP_Chain + pmtx(j), "a+⋅", NP_Chain + pmtx(j + 1)
        thempo += -1im * eff_gs[j], "a+⋅", NP_Chain + pmtx(j), "a-⋅", NP_Chain + pmtx(j + 1)

        thempo += +1im * eff_gs[j], "⋅a-", NP_Chain + pmtx(j), "⋅a+", NP_Chain + pmtx(j + 1)
        thempo += +1im * eff_gs[j], "⋅a+", NP_Chain + pmtx(j), "⋅a-", NP_Chain + pmtx(j + 1)
    end

    #################################

    #Primary chain - MC interaction
    #Debug
    for j in 1:MC_N
        thempo += -1im * eff_coups[j], "a-⋅", NP_Chain, "a+⋅", NP_Chain + pmtx(j)
        thempo += -1im * conj(eff_coups[j]), "a+⋅", NP_Chain, "a-⋅", NP_Chain + pmtx(j)

        thempo += +1im * eff_coups[j], "⋅a-", NP_Chain, "⋅a+", NP_Chain + pmtx(j)
        thempo += +1im * conj(eff_coups[j]), "⋅a+", NP_Chain, "⋅a-", NP_Chain + pmtx(j)
    end
    #################################

    #Lindblad terms
    #occhio al valore di gamma...
    for j in 1:MC_N
        thempo += 2 * gammas[j], "Lindb+", NP_Chain + pmtx(j)
    end
    #################################

    return MPO(thempo, sysenv)
end

function createObs(lookat)
    vobs = []
    for a in lookat
        push!(vobs, opPos(a[1], a[2]))
    end

    return vobs
end

function stretchBondDim(state::MPS, extDim::Int64)
    psiExt = copy(state)
    NN = length(psiExt)
    for n in 1:(NN - 1)
        a = commonind(psiExt[n], psiExt[n + 1])
        tagsa = tags(a)
        add_indx = Index(extDim; tags=tagsa)
        psiExt[n] = psiExt[n] * delta(a, add_indx)
        psiExt[n + 1] = psiExt[n + 1] * delta(a, add_indx)
    end
    #println("Overlap <original|extended> states: ", dot(state,psiExt));
    return psiExt, dot(state, psiExt)
end
