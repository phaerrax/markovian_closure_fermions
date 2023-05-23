using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS
using KrylovKit

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]

    # Input: chain stub parameters
    # ----------------------------
    chain_length = parameters["chain_length"]
    thermofield_coefficients = readdlm(
        parameters["thermofield_coefficients"], ',', Float64; skipstart=1
    )
    coups = thermofield_coefficients[:, 1]
    freqs = thermofield_coefficients[:, 3]

    # Input: closure parameters
    # -------------------------
    Ω = 0
    K = 0.5

    α = readdlm(parameters["MC_alphas"]) # α[l,1] = Re(αₗ) and α[l,2] = Im(αₗ)...
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    mcω = @. Ω - 2K * α[:, 2]
    mcγ = @. -4K * α[:, 1]
    mcg = @. -2K * β[:, 2]
    mcζ = @. K * (w[:, 1] + im * w[:, 2])
    closure_length = length(mcω)

    sites = siteinds("vS=1/2", system_length + chain_length + closure_length)
    psi = MPS(sites, [system_initstate; repeat(["Dn"], chain_length + closure_length)])

    # Unitary part of master equation
    # -------------------------------
    # -i [H, ρ] = -i H ρ + i ρ H
    ℓ₁ = OpSum()

    # System Hamiltonian
    # (We assume system_length == 1 for now...)
    ℓ₁ += eps * gkslcommutator("N", 1)

    if chain_length > 0
        # System-chain interaction:
        ℓ₁ += -coups[1] * gkslcommutator("σy", 1, "σx", 2)

        # Hamiltonian of the chain stub:
        # - local frequency terms
        for j in 1:chain_length
            ℓ₁ += freqs[j] * gkslcommutator("N", system_length + j)
        end
        # - coupling between sites
        for j in 1:(chain_length - 1)
            # coups[1] is the coupling coefficient between the open system and the first
            # site of the chain; we don't need it here.
            site1 = system_length + j
            site2 = system_length + j + 1
            ℓ₁ += coups[j + 1] * gkslcommutator("σ+", site1, "σ-", site2)
            ℓ₁ += coups[j + 1] * gkslcommutator("σ-", site1, "σ+", site2)
        end
    end

    ℓ₂ = OpSum()
    # Hamiltonian of the closure:
    # - local frequency terms
    for k in 1:closure_length
        pmsite = system_length + chain_length + k
        ℓ₂ += mcω[k] * gkslcommutator("N", pmsite)
    end
    # - coupling between pseudomodes
    for k in 1:(closure_length - 1)
        pmode_site1 = system_length + chain_length + k
        pmode_site2 = system_length + chain_length + k + 1
        ℓ₂ += mcg[k] * gkslcommutator("σ-", pmode_site1, "σ+", pmode_site2)
        ℓ₂ += mcg[k] * gkslcommutator("σ+", pmode_site1, "σ-", pmode_site2)
    end
    # - coupling between the end of the chain stub and each pseudomode
    for j in 1:closure_length
        # Here come the Pauli strings...
        chainedge_site = system_length + chain_length
        pmode_site = system_length + chain_length + j
        ps_length = pmode_site - chainedge_site - 1 # == j-1

        paulistring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        ℓ₂ +=
            (-1)^ps_length *
            mcζ[j] *
            gkslcommutator(zip(paulistring, chainedge_site:pmode_site)...)

        paulistring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        ℓ₂ +=
            (-1)^ps_length *
            conj(mcζ[j]) *
            gkslcommutator(zip(paulistring, chainedge_site:pmode_site)...)
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    for j in 1:closure_length
        pmode_site = system_length + chain_length + j
        # a ρ a†
        opstring = [repeat(["σz⋅ * ⋅σz"], pmode_site - 1); "σ-⋅ * ⋅σ+"]
        ℓ₂ += (mcγ[j], collect(Iterators.flatten(zip(opstring, 1:pmode_site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ₂ += -0.5mcγ[j], "N⋅", pmode_site
        ℓ₂ += -0.5mcγ[j], "⋅N", pmode_site
    end

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

    ω⃗ = [1, 2]
    f⃗ = [t -> cos(ω * t) for ω in ω⃗]

    ℋ⃗₀ = [ℓ₁, ℓ₂]

    H⃗₀ = [MPO(ℋᵢ, sites) for ℋᵢ in ℋ⃗₀]

    krylov_kwargs = (;
        ishermitian=false, krylovdim=parameters["krylov_dim"], tol=parameters["exp_tol"]
    )

    #  Specific solver function for time-dependent TDVP.
    #  We don't need an "inner" function such as in exponentiate_solver above because...
    function time_dependent_solver(
        H::TimeDependentSum, time_step, ψ₀; current_time=0.0, outputlevel=0, kwargs...
    )
        @debug "In Krylov solver, current_time = $current_time, time_step = $time_step"
        ψₜ, info = exponentiate(H(current_time), time_step, ψ₀; kwargs...)
        return ψₜ, info
    end

    function td_solver(Hs::ProjMPOSum, time_step, ψ₀; kwargs...)
        # Questa è la funzione che viene innestata in tdvp_site_update!.
        # Le vengono forniti gli argomenti (H, time_step, psi; current_time); con H costruiamo
        # l'oggetto TimeDependentSum.
        # A sua volta, chiama time_dependent_solver con un TimeDependentSum, che è definito in
        # TimeEvoVecMPS/src/tdvp_step.jl. Qui attacchiamo anche i kwargs specifici di exponentiate,
        # che vengono passati alla funzione dal time_dependent_solver "più interno".
        return time_dependent_solver(
            -im * TimeDependentSum(f⃗, Hs), time_step, ψ₀; krylov_kwargs..., kwargs...
        )
    end

    growMPS!(psi, parameters["max_bond"])

    @info "Starting simulation."
    tdvp1vec!(
        td_solver,
        psi,
        H⃗₀,
        timestep,
        tmax,
        sites;
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
        normalize=false,
        convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
        max_bond=parameters["max_bond"],
    )
end
