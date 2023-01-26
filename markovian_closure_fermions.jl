using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

include("./TDVP_lib_VecRho.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]
    delta = parameters["sys_coup"]

    # Input: chain stub parameters
    coups = readdlm(parameters["chain_freqs"]) # Coupling constants
    freqs = readdlm(parameters["chain_coups"]) # Frequencies
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    omega = parameters["omegaInf"]
    alphas_MC = readdlm(parameters["MC_alphas"]) # Dissipation constants
    betas_MC = readdlm(parameters["MC_betas"]) # Coupling between pseudomodes 
    coups_MC = readdlm(parameters["MC_coups"]) # Coupling between pseudomodes and chain edge
    gammas = omega * alphas_MC[:, 1]
    eff_freqs = [omega + 0.0 for g in gammas] # ???
    eff_gs = omega * betas_MC[:, 2]
    eff_coups = omega / 2 * (coups_MC[:, 1] + im * coups_MC[:, 2])
    closure_length = length(gammas)

    perm = get(parameters, "perm", nothing)
    if !isnothing(perm)
        if length(perm) != closure_length
            println("The provided permutation is not correct")
        end
        pmtx = Permutation(perm)
        @show pmtx
    else
        # Identity permutation
        pmtx = Permutation(collect(1:closure_length))
        @show pmtx
    end

    sites = siteinds("HvS=1/2", system_length + chain_length + closure_length)
    psi0 = productMPS(
        sites, [system_initstate; repeat(["Dn"], chain_length + closure_length)]
    )

    # Unitary part of master equation
    # -------------------------------
    # -i [H, ρ] = -i H ρ + i ρ H
    ℓ = OpSum()

    # System Hamiltonian
    # (We assume system_length == 1 for now...)
    ℓ += -im * eps, "N⋅", 1
    ℓ += +im * eps, "⋅N", 1

    #ℓ += -im * delta, "σx⋅", 1
    #ℓ +=  im * delta, "⋅σx", 1

    # System-chain interaction:
    ℓ += -im * coups[1], "σ+⋅", 1, "σ-⋅", 2
    ℓ += -im * coups[1], "σ-⋅", 1, "σ+⋅", 2
    ℓ += +im * coups[1], "⋅σ+", 1, "⋅σ-", 2
    ℓ += +im * coups[1], "⋅σ-", 1, "⋅σ+", 2

    # Hamiltonian of the chain stub:
    # - local frequency terms
    for j in system_length .+ (1:chain_length)
        ℓ += -im * freqs[j - 1], "N⋅", j
        ℓ += +im * freqs[j - 1], "⋅N", j
    end
    # - coupling between sites
    for j in system_length .+ (1:(chain_length - 1))
        ℓ += -1im * coups[j], "σ+⋅", j, "σ-⋅", j + 1
        ℓ += -1im * coups[j], "σ-⋅", j, "σ+⋅", j + 1
        ℓ += +1im * coups[j], "⋅σ+", j, "⋅σ-", j + 1
        ℓ += +1im * coups[j], "⋅σ-", j, "⋅σ+", j + 1
    end

    # Hamiltonian of the closure:
    # - local frequency terms
    for k in 1:closure_length
        j = system_length + chain_length + pmtx(k)
        ℓ += -im * eff_freqs[k], "N⋅", j
        ℓ += +im * eff_freqs[k], "⋅N", j
    end
    # - coupling between pseudomodes
    for k in 1:(closure_length - 1)
        j1 = system_length + chain_length + pmtx(k)
        j2 = system_length + chain_length + pmtx(k + 1)
        ℓ += -im * eff_gs[k], "σ-⋅", j1, "σ+⋅", j2
        ℓ += -im * eff_gs[k], "σ+⋅", j1, "σ-⋅", j2
        ℓ += +im * eff_gs[k], "⋅σ-", j1, "⋅σ+", j2
        ℓ += +im * eff_gs[k], "⋅σ+", j1, "⋅σ-", j2
    end
    # - coupling between the end of the chain stub and each pseudomode
    #
    # Before we can proceed, we need to convert the AutoMPO we built until now into a
    # concrete MPO. The following MPOs, in fact, are tensor products of more than
    # two non-trivial operators, so they cannot be built using the AutoMPO syntax.
    # Instead, we define them with a standard MPO constructor and add them one by one.
    L = MPO(ℓ, sites)
    for j in 1:closure_length
        # Here come the Pauli strings...
        N = length(sites)
        pstring1 = ["σ+"; repeat(["σz"], j - 1); "σ-"]
        pstring2 = ["σ-"; repeat(["σz"], j - 1); "σ+"]
        L +=
            -im *
            (-1)^(j - 1) *
            (
                MPO(
                    sites,
                    [
                        repeat(["Id"], system_length + chain_length - 1)
                        pstring1 .* "⋅"
                        repeat(["Id"], N - chain_length - system_length - j)
                    ],
                ) - MPO(
                    sites,
                    [
                        repeat(["Id"], system_length + chain_length - 1)
                        "⋅" .* pstring1
                        repeat(["Id"], N - chain_length - system_length - j)
                    ],
                )
            )

        L +=
            -im *
            (-1)^(j - 1) *
            (
                MPO(
                    sites,
                    [
                        repeat(["Id"], system_length + chain_length - 1)
                        pstring2 .* "⋅"
                        repeat(["Id"], N - chain_length - system_length - j)
                    ],
                ) - MPO(
                    sites,
                    [
                        repeat(["Id"], system_length + chain_length - 1)
                        "⋅" .* pstring2
                        repeat(["Id"], N - chain_length - system_length - j)
                    ],
                )
            )
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    for j in 1:closure_length
        N = length(sites)
        # a ρ a†
        pstring1 = [
            repeat(["σz⋅"], system_length + chain_length + j - 1)
            "σ-⋅"
            repeat(["Id"], N - system_length - chain_length - j)
        ]
        pstring2 = [
            repeat(["⋅σz"], system_length + chain_length + j - 1)
            "⋅σ+"
            repeat(["Id"], N - system_length - chain_length - j)
        ]
        # The minus signs collected from the -σz factors all cancel out.
        M = replaceprime(contract(MPO(sites, pstring1)', MPO(sites, pstring2)), 2 => 1)
        # In `contract(A::MPO, B::MPO)`, MPOs A and B have the same site indices. The
        # indices of the MPOs in the contraction are taken literally, and therefore they
        # should only share one site index per site so the contraction results in an MPO.
        # (The two MPOs commute.)

        # -0.5 (a† a ρ + ρ a† a)
        M +=
            -0.5 * (
                MPO(sites, [k == j ? "N⋅" : "Id" for k in 1:N]) + MPO(sites, [k == j ? "⋅N" : "Id" for k in 1:N])
            )

        # Multiply by the site's damping coefficient and add it to L.
        L += gammas[j] * M
    end

    # Define some quantities that must be observed.
    # Firstly, the norm, σˣ and σᶻ on the system site.
    obs = [["Norm", 1], ["vecσx", 1], ["vecσz", 1]]
    # Then, the occupation number in some sites of the chain...
    for i in 10:10:(system_length + chain_length)
        push!(obs, ["vecN", i + 1])
    end
    # ...and in each pseudomode.
    for i in (system_length + chain_length) .+ (1:closure_length)
        push!(obs, ["vecN", i + 1])
    end

    # Enlarge the bond dimensions so that TDVP1 has the possibility to grow
    # the number of singular values between the bonds.
    psi, overlap = stretchBondDim(psi0, 20)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    # Create a LocalPosMeasurementCallback where we append to the names of each operator a
    # subscript with the site to which it refers.
    cb = LocalPosMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )
    # `vobs` isa Vector{opPos}: each element is a couple of elements, the first is an operator
    # and the second a position (aka a site along the chain).

    tdvp1vec!(
        psi,
        L,
        timestep,
        tmax;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        exp_tol=parameters["exp_tol"],
        krylovdim=parameters["krylov_dim"],
        store_psi0=false,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
    )
end
