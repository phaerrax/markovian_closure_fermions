using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

include("./TDVP_lib_VecRho.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]
    delta = parameters["sys_coup"]

    # Input: chain stub parameters
    # ----------------------------
    tedopa_coefficients = readdlm(
        parameters["tedopa_coefficients"], ',', Float64; skipstart=1
    )
    coups = tedopa_coefficients[:, 1]
    freqs = tedopa_coefficients[:, 2]
    chain_length = parameters["chain_length"]

    # Input: closure parameters
    # -------------------------
    # The parameters given in the files refer to the correlation function c(t) = J₀(t)+J₂(t)
    # which corresponds to the standard semicircle spectral density
    #   j(x) = 2/pi * sqrt(1-x²).
    # A general semicircle is given by
    #   J(x) = 1/2pi * sqrt( (2K - Ω + x) * (2K + Ω - x) ) =
    #        = 1/2pi * sqrt( (x - ωmin) * (ωmax - x) ) =
    #        = K/2 * j( (x - Ω) / 2K ),
    # so j is J with K = 1/2 and Ω = 0; their correlation functions c(t) and C(t) are
    # related by
    #   C(t) = K^2 * exp(-iΩt) * c(2Kt).
    # In a generic TEDOPA environment with domain (ωmin, ωmax), the coefficients K and Ω
    # come from the asymptotic limits of the chain coefficients, and
    #   Ω = (ωmax + ωmin) / 2,
    #   K = (ωmax - ωmin) / 4.
    #
    # If the matrix
    #       ⎛ α₁ β₁ 0  ⋯  0  ⎞
    #       ⎜ β₁ α₂ β₂ ⋯  0  ⎟
    #   M = ⎜ 0  β₂ α₃ ⋯  0  ⎟
    #       ⎜ ⋮  ⋮  ⋮  ⋱  ⋮  ⎟
    #       ⎝ 0  0  0  ⋯  αₙ ⎠
    # and the vector w fit the relation
    #   w† exp(-itM) w ≈ c(t)
    # i.e. they solve the "standard semicircle" j, then the rescaled coefficients
    #   ωⱼ = Ω - 2K Im(αⱼ)
    #   γⱼ = -4K Re(αⱼ)
    #   gⱼ = -2K Im(βⱼ)
    #   ζⱼ = K wⱼ
    # describe the pseudomode surrogate environment from the spectral density J.
    #
    Ω = parameters["omegaInf"]
    K = Ω / 2
    # We assume that ωmin = 0, therefore Ω = ωmax/2 and K = ωmax/4.

    α = readdlm(parameters["MC_alphas"]) # α[l,1] = Re(αₗ) and α[l,2] = Im(αₗ)...
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    mcω = @. Ω - 2K * α[:, 2]
    mcγ = @. -4K * α[:, 1]
    mcg = @. -2K * β[:, 2]
    mcζ = @. K * (w[:, 1] + im * w[:, 2])
    closure_length = length(mcω)

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

    if chain_length > 0
        # System-chain interaction:
        if lowercase(parameters["interaction_type"]) == "xx"
            ℓ += -im * coups[1], "σx⋅", 1, "σx⋅", 2
            ℓ += +im * coups[1], "⋅σx", 1, "⋅σx", 2
        elseif lowercase(parameters["interaction_type"]) == "exchange"
            ℓ += -im * coups[1], "σ+⋅", 1, "σ-⋅", 2
            ℓ += -im * coups[1], "σ-⋅", 1, "σ+⋅", 2
            ℓ += +im * coups[1], "⋅σ+", 1, "⋅σ-", 2
            ℓ += +im * coups[1], "⋅σ-", 1, "⋅σ+", 2
        else
            throw(
                error("Unrecognized interaction type. Please use \"xx\" or \"exchange\".")
            )
        end

        # Hamiltonian of the chain stub:
        # - local frequency terms
        for j in 1:chain_length
            ℓ += -im * freqs[j], "N⋅", system_length + j
            ℓ += +im * freqs[j], "⋅N", system_length + j
        end
        # - coupling between sites
        for j in 1:(chain_length - 1)
            # coups[1] is the coupling coefficient between the open system and the first
            # site of the chain; we don't need it here.
            ℓ += -im * coups[j + 1], "σ+⋅", system_length + j, "σ-⋅", system_length + j + 1
            ℓ += -im * coups[j + 1], "σ-⋅", system_length + j, "σ+⋅", system_length + j + 1
            ℓ += +im * coups[j + 1], "⋅σ+", system_length + j, "⋅σ-", system_length + j + 1
            ℓ += +im * coups[j + 1], "⋅σ-", system_length + j, "⋅σ+", system_length + j + 1
        end
    end

    # Hamiltonian of the closure:
    # - local frequency terms
    for k in 1:closure_length
        j = system_length + chain_length + pmtx(k)
        ℓ += -im * mcω[k], "N⋅", j
        ℓ += +im * mcω[k], "⋅N", j
    end
    # - coupling between pseudomodes
    for k in 1:(closure_length - 1)
        j1 = system_length + chain_length + pmtx(k)
        j2 = system_length + chain_length + pmtx(k + 1)
        ℓ += -im * mcg[k], "σ-⋅", j1, "σ+⋅", j2
        ℓ += -im * mcg[k], "σ+⋅", j1, "σ-⋅", j2
        ℓ += +im * mcg[k], "⋅σ-", j1, "⋅σ+", j2
        ℓ += +im * mcg[k], "⋅σ+", j1, "⋅σ-", j2
    end
    # - coupling between the end of the chain stub and each pseudomode
    for j in 1:closure_length
        # Here come the Pauli strings...
        chainedge_site = system_length + chain_length
        pmode_site = system_length + chain_length + j
        ps_length = pmode_site - chainedge_site - 1 # == j-1

        opstring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        ℓ += (
            -im * (-1)^ps_length * mcζ[j],
            collect(Iterators.flatten(zip(opstring .* "⋅", chainedge_site:pmode_site)))...,
        )
        # This becomes the tuple
        #   -i(-1)ʲ⁻¹ζⱼ, "σ+", Nₑ, "σz", Nₑ+1, ..., "σz", m-1, "σ-", m
        # where m = Nₛ+Nₑ+j is the site of the pseudomode.
        #
        # As a check:
        #   chainedge_site:pmode_site ==
        #   (system_length + chain_length:system_length + chain_length + j) ==
        #   (system_length + chain_length) .+ (0:j)
        # so it contains j+1 elements, which is also the number of elements in the vector
        #   ["σ+"; repeat(["σz"], ps_length); "σ-"].
        # Now we do the same for the remaining terms.
        ℓ += (
            im * (-1)^ps_length * mcζ[j],
            collect(Iterators.flatten(zip("⋅" .* opstring, chainedge_site:pmode_site)))...,
        )

        opstring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        ℓ += (
            -im * (-1)^ps_length * conj(mcζ[j]),
            collect(Iterators.flatten(zip(opstring .* "⋅", chainedge_site:pmode_site)))...,
        )
        ℓ += (
            im * (-1)^ps_length * conj(mcζ[j]),
            collect(Iterators.flatten(zip("⋅" .* opstring, chainedge_site:pmode_site)))...,
        )
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    for j in 1:closure_length
        pmode_site = system_length + chain_length + j
        # a ρ a†
        opstring = [repeat(["σz⋅ * ⋅σz"], pmode_site - 1); "σ-⋅ * ⋅σ+"]
        ℓ += (mcγ[j], collect(Iterators.flatten(zip(opstring, 1:pmode_site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ += -0.5mcγ[j], "N⋅", pmode_site
        ℓ += -0.5mcγ[j], "⋅N", pmode_site
    end
    L = MPO(ℓ, sites)

    # Enlarge the bond dimensions so that TDVP1 has the possibility to grow
    # the number of singular values between the bonds.
    psi, overlap = stretchBondDim(psi0, parameters["max_bond"])

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    obs = []
    oblist = parameters["observables"]
    for key in keys(oblist)
        foreach(i -> push!(obs, [key, i]), oblist[key])
    end

    cb = LocalPosVecMeasurementCallback(
        createObs(obs), sites, parameters["ms_stride"] * timestep
    )

    tdvp1vec!(
        psi,
        L,
        timestep,
        tmax,
        sites;
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
