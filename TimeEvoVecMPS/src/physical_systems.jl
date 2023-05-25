export Closure, closure
export freq,
       innercoup,
       outercoup,
       damp,
       freqs,
       innercoups,
       outercoups,
       damps

export exchange_interaction, spin_chain, closure_op, filled_closure_op
export defineSystem, createMPO, createMPO2MC, createMPOVecRho

struct Closure
    frequency
    innercoupling
    outercoupling
    damping
    function Closure(ω::Vector{<:Real}, γ::Vector{<:Real}, g::Vector{<:Real}, ζ::Vector{<:Complex})
        if (
            length(ω) - 1 != length(g) ||
            length(ω) != length(γ) ||
            length(ω) != length(ζ)
        )
            error("Lengths of input parameters do not match.")
        end
        return new(ω, g, ζ, γ)
    end
end
function closure(
    Ω::Number, K::Number, α::Matrix{<:Real}, β::Matrix{<:Real}, w::Matrix{<:Real}
)
    return closure(
        Ω, K, α[:, 1] .+ im .* α[:, 2], β[:, 1] .+ im .* β[:, 2], w[:, 1] .+ im .* w[:, 2]
    )
end
function closure(
    Ω::Number, K::Number, α::Vector{<:Complex}, β::Vector{<:Complex}, w::Vector{<:Complex}
)
    frequency = @. Ω - 2K * imag(α)
    damping = @. -4K * real(α)
    innercoupling = @. -2K * imag(β)
    outercoupling = @. K * w
    return Closure(frequency, damping, innercoupling, outercoupling)
end

Base.length(mc::Closure) = Base.length(mc.frequency)

freqs(mc::Closure) = mc.frequency
innercoups(mc::Closure) = mc.innercoupling
outercoups(mc::Closure) = mc.outercoupling
damps(mc::Closure) = mc.damping

freq(mc::Closure, j::Int) = mc.frequency[j]
innercoup(mc::Closure, j::Int) = mc.innercoupling[j]
outercoup(mc::Closure, j::Int) = mc.outercoupling[j]
damp(mc::Closure, j::Int) = mc.damping[j]

"""
    closure_op(mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int)

Return an OpSum object encoding the Markovian closure operators with parameters given by
`mc`, on sites `sites`, linked to the main TEDOPA/thermofield chain on site
`chain_edge_site`.
This closure replaces a chain starting from an empty state.
"""
function closure_op(
    ::SiteType"vS=1/2", mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int
)
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freq(mc, j) * gkslcommutator("N", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        jws = jwstring(; start=site1, stop=site2)
        ℓ +=
            innercoup(mc, j) * (
                gkslcommutator("σ-", site1, jws..., "σ+", site2) +
                gkslcommutator("σ+", site1, jws..., "σ-", site2)
            )
    end
    for (j, site) in enumerate(sitenumber.(sites))
        jws = jwstring(; start=chain_edge_site, stop=site)
        ℓ += (
            outercoup(mc, j) * gkslcommutator("σ+", chain_edge_site, jws..., "σ-", site) +
            conj(outercoup(mc, j)) *
            gkslcommutator("σ-", chain_edge_site, jws..., "σ+", site)
        )
    end

    for (j, site) in enumerate(sitenumber.(sites))
        # a ρ a†
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "σ-⋅ * ⋅σ+"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ += -0.5damp(mc, j), "N⋅", site
        ℓ += -0.5damp(mc, j), "⋅N", site
    end
    return ℓ
end

function closure_op(::SiteType"vElectron", mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int)
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freq(mc, j) * gkslcommutator("Ntot", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        # Add as many "F" string operator as needed.
        jws = jwstring(; start=site1, stop=site2)
        ℓ +=
            innercoup(mc, j) * (
                gkslcommutator("Aup†F", site1, jws..., "Aup", site2) -
                gkslcommutator("AupF", site1, jws..., "Aup†", site2) +
                gkslcommutator("Adn†", site1, jws..., "FAdn", site2) -
                gkslcommutator("Adn", site1, jws..., "FAdn†", site2)
            )
    end
    for (j, site) in enumerate(sitenumber.(sites))
        # c↑ᵢ† c↑ᵢ₊ₙ = a↑ᵢ† Fᵢ Fᵢ₊₁ ⋯ Fᵢ₊ₙ₋₁ a↑ᵢ₊ₙ
        # c↑ᵢ₊ₙ† c↑ᵢ = -a↑ᵢ Fᵢ Fᵢ₊₁ ⋯ Fᵢ₊ₙ₋₁ a↑ᵢ₊ₙ†
        # c↓ᵢ† c↓ᵢ₊ₙ = a↓ᵢ† Fᵢ₊₁ Fᵢ₊₂ ⋯ Fᵢ₊ₙ a↓ᵢ₊ₙ
        # c↓ᵢ₊ₙ† c↓ᵢ = -a↓ᵢ Fᵢ₊₁ Fᵢ₊₂ ⋯ Fᵢ₊ₙ a↓ᵢ₊ₙ†

        jws = jwstring(; start=chain_edge_site, stop=site)
        # ζⱼ c↑₀† c↑ⱼ (0 = chain edge, j = pseudomode)
        ℓ +=
            outercoup(mc, j) * gkslcommutator("Aup†F", chain_edge_site, jws..., "Aup", site)
        # conj(ζⱼ) c↑ⱼ† c↑₀
        ℓ +=
            -conj(outercoup(mc, j)) *
            gkslcommutator("AupF", chain_edge_site, jws..., "Aup†", site)
        # ζⱼ c↓₀† c↓ⱼ
        ℓ +=
            outercoup(mc, j) * gkslcommutator("Adn†", chain_edge_site, jws..., "FAdn", site)
        # conj(ζⱼ) c↓ⱼ† c↓₀
        ℓ +=
            -conj(outercoup(mc, j)) *
            gkslcommutator("Adn", chain_edge_site, jws..., "FAdn†", site)
    end

    # Dissipative part
    for (j, site) in enumerate(sitenumber.(sites))
        # c↑ₖ ρ c↑ₖ† = F₁ ⋯ Fₖ₋₁ a↑ₖ ρ a↑ₖ† Fₖ₋₁ ⋯ F₁
        # Remember that:
        # • Fⱼ = (1 - 2 N↑ₖ) (1 - 2 N↓ₖ);
        # • Fⱼ and aₛ,ₖ commute only on different sites;
        # • {a↓ₖ, a↓ₖ†} = {a↑ₖ, a↑ₖ†} = 1;
        # • Fₖ anticommutes with a↓ₖ, a↓ₖ†, a↑ₖ and a↑ₖ†.
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Aup⋅ * ⋅Aup†"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # c↓ₖ ρ c↓ₖ† = F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ ρ a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "FAdn⋅ * ⋅Adn†F"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)

        # -½ (c↑ₖ† c↑ₖ ρ + ρ c↑ₖ† c↑ₖ) = -½ (a↑ₖ† a↑ₖ ρ + ρ a↑ₖ† a↑ₖ)
        ℓ += -0.5damp(mc, j), "Nup⋅", site
        ℓ += -0.5damp(mc, j), "⋅Nup", site
        # -½ (c↓ₖ† c↓ₖ ρ + ρ c↓ₖ† c↓ₖ) = -½ (a↓ₖ† a↓ₖ ρ + ρ a↓ₖ† a↓ₖ)
        ℓ += -0.5damp(mc, j), "Ndn⋅", site
        ℓ += -0.5damp(mc, j), "⋅Ndn", site
    end
    return ℓ
end

"""
    filled_closure_op(mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int)

Return an OpSum object encoding the Markovian closure operators with parameters given by
`mc`, on sites `sites`, linked to the main TEDOPA/thermofield chain on site
`chain_edge_site`.
This closure replaces a chain starting from a completely filled state.
"""
function filled_closure_op(
    ::SiteType"vS=1/2", mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int
)
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freq(mc, j) * gkslcommutator("N", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        jws = jwstring(; start=site1, stop=site2)
        ℓ +=
            innercoup(mc, j) * (
                gkslcommutator("σ-", site1, jws..., "σ+", site2) +
                gkslcommutator("σ+", site1, jws..., "σ-", site2)
            )
    end
    for (j, site) in enumerate(sitenumber.(sites))
        jws = jwstring(; start=chain_edge_site, stop=site)
        ℓ += (
            outercoup(mc, j) * gkslcommutator("σ+", chain_edge_site, jws..., "σ-", site) +
            conj(outercoup(mc, j)) *
            gkslcommutator("σ-", chain_edge_site, jws..., "σ+", site)
        )
    end

    for (j, site) in enumerate(sitenumber.(sites))
        # a† ρ a
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "σ+⋅ * ⋅σ-"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # -0.5 (a a† ρ + ρ a a†)
        ℓ += 0.5damp(mc, j), "N⋅", site
        ℓ += 0.5damp(mc, j), "⋅N", site
        ℓ += -damp(mc, j), "Id", site
    end
    return ℓ
end

function filled_closure_op(::SiteType"vElectron", mc::Closure, sites::Vector{<:Index}, chain_edge_site::Int)
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freq(mc, j) * gkslcommutator("Ntot", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        # Add as many "F" string operator as needed.
        jws = jwstring(; start=site1, stop=site2)
        ℓ +=
            innercoup(mc, j) * (
                gkslcommutator("Aup†F", site1, jws..., "Aup", site2) -
                gkslcommutator("AupF", site1, jws..., "Aup†", site2) +
                gkslcommutator("Adn†", site1, jws..., "FAdn", site2) -
                gkslcommutator("Adn", site1, jws..., "FAdn†", site2)
            )
    end
    for (j, site) in enumerate(sitenumber.(sites))
        jws = jwstring(; start=chain_edge_site, stop=site)
        # ζⱼ c↑₀† c↑ⱼ (0 = chain edge, j = pseudomode)
        ℓ +=
            outercoup(mc, j) * gkslcommutator("Aup†F", chain_edge_site, jws..., "Aup", site)
        # conj(ζⱼ) c↑ⱼ† c↑₀
        ℓ +=
            -conj(outercoup(mc, j)) *
            gkslcommutator("AupF", chain_edge_site, jws..., "Aup†", site)
        # ζⱼ c↓₀† c↓ⱼ
        ℓ +=
            outercoup(mc, j) * gkslcommutator("Adn†", chain_edge_site, jws..., "FAdn", site)
        # conj(ζⱼ) c↓ⱼ† c↓₀
        ℓ +=
            -conj(outercoup(mc, j)) *
            gkslcommutator("Adn", chain_edge_site, jws..., "FAdn†", site)
    end

    # Dissipative part
    for (j, site) in enumerate(sitenumber.(sites))
        # c↑ₖ† ρ c↑ₖ = a↑ₖ† Fₖ₋₁ ⋯ F₁ ρ F₁ ⋯ Fₖ₋₁ a↑ₖ
        # Remember that:
        # • Fⱼ = (1 - 2 N↑ₖ) (1 - 2 N↓ₖ);
        # • Fⱼ and aₛ,ₖ commute only on different sites;
        # • {a↓ₖ, a↓ₖ†} = {a↑ₖ, a↑ₖ†} = 1;
        # • Fₖ anticommutes with a↓ₖ, a↓ₖ†, a↑ₖ and a↑ₖ†.
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Aup†⋅ * ⋅Aup"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)
        # c↓ₖ† ρ c↓ₖ = a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁ ρ F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ
        opstring = [repeat(["F⋅ * ⋅F"], site - 1); "Adn†F⋅ * ⋅FAdn"]
        ℓ += (damp(mc, j), collect(Iterators.flatten(zip(opstring, 1:site)))...)

        # c↑ₖ c↑ₖ† = F₁ ⋯ Fₖ₋₁ a↑ₖ a↑ₖ† Fₖ₋₁ ⋯ F₁ =
        #          = F₁² ⋯ Fₖ₋₁² a↑ₖ a↑ₖ† =
        #          = a↑ₖ a↑ₖ† =
        #          = 1 - N↑ₖ
        ℓ += -damp(mc, j), "Id", site
        ℓ += 0.5damp(mc, j), "Nup⋅", site
        ℓ += 0.5damp(mc, j), "⋅Nup", site
        # c↓ₖ c↓ₖ† = F₁ ⋯ Fₖ₋₁ Fₖa↓ₖ a↓ₖ†Fₖ Fₖ₋₁ ⋯ F₁ =
        #          = F₁² ⋯ Fₖ₋₁² Fₖa↓ₖ a↓ₖ†Fₖ =
        #          = Fₖa↓ₖ a↓ₖ†Fₖ =
        #          = Fₖ² a↓ₖ a↓ₖ† =
        #          = 1 - N↓ₖ
        ℓ += -damp(mc, j), "Id", site
        ℓ += 0.5damp(mc, j), "Ndn⋅", site
        ℓ += 0.5damp(mc, j), "⋅Ndn", site
    end
    return ℓ
end

"""
    spin_chain(freqs, coups, sites::Vector{<:Index})

Return an OpSum object encoding the Hamiltonian part ``-i[H, –]`` of the GKSL equation
for a spin chain of frequencies `freqs` and coupling constants `coups`, on `sites`.
"""
function spin_chain(::SiteType"vS=1/2", freqs, coups, sites::Vector{<:Index})
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freqs[j] * gkslcommutator("N", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        jws = jwstring(; start=site1, stop=site2)
        ℓ +=
            coups[j + 1] * (
                gkslcommutator("σ+", site1, jws..., "σ-", site2) +
                gkslcommutator("σ-", site1, jws..., "σ+", site2)
            )
    end
    return ℓ
end

function spin_chain(::SiteType"vElectron", freqs, coups, sites::Vector{<:Index})
    ℓ = OpSum()
    for (j, site) in enumerate(sitenumber.(sites))
        ℓ += freqs[j] * gkslcommutator("Ntot", site)
    end
    for (j, (site1, site2)) in enumerate(partition(sitenumber.(sites), 2, 1))
        jws = jwstring(; start=site1, stop=site2)
        ℓ += +coups[j + 1] * gkslcommutator("Aup†F", site1, jws..., "Aup", site2)
        ℓ += -coups[j + 1] * gkslcommutator("AupF", site1, jws..., "Aup†", site2)
        ℓ += +coups[j + 1] * gkslcommutator("Adn†", site1, jws..., "FAdn", site2)
        ℓ += -coups[j + 1] * gkslcommutator("Adn", site1, jws..., "FAdn†", site2)
    end
    return ℓ
end

function exchange_interaction(::SiteType"vS=1/2", s1::Index, s2::Index)
    site1 = sitenumber(s1)
    site2 = sitenumber(s2)

    ℓ = OpSum()
    jws = jwstring(; start=site1, stop=site2)
    ℓ += (
        gkslcommutator("σ+", site1, jws..., "σ-", site2) +
        gkslcommutator("σ-", site1, jws..., "σ+", site2)
    )
    return ℓ
end

function exchange_interaction(::SiteType"vElectron", s1::Index, s2::Index)
    site1 = sitenumber(s1)
    site2 = sitenumber(s2)
    # c↑ᵢ† c↑ᵢ₊₁ + c↑ᵢ₊₁† c↑ᵢ + c↓ᵢ† c↓ᵢ₊₁ + c↓ᵢ₊₁† c↓ᵢ =
    # a↑ᵢ† Fᵢ a↑ᵢ₊₁ - a↑ᵢ Fᵢ a↑ᵢ₊₁† + a↓ᵢ† Fᵢ₊₁ a↓ᵢ₊₁ - a↓ᵢ Fᵢ₊₁ a↓ᵢ₊₁†
    ℓ = OpSum()
    jws = jwstring(; start=site1, stop=site2)
    ℓ += (
        gkslcommutator("Aup†F", site1, jws..., "Aup", site2) -
        gkslcommutator("AupF", site1, jws..., "Aup†", site2) +
        gkslcommutator("Adn†", site1, jws..., "FAdn", site2) -
        gkslcommutator("Adn", site1, jws..., "FAdn†", site2)
    )
    return ℓ
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
