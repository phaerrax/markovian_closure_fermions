using ITensors
using DelimitedFiles
using PseudomodesTTEDOPA
using TimeEvoVecMPS

include("TDVP_lib_VecRho.jl")

let
    parameters = load_pars(ARGS[1])

    # Input: system parameters
    # ------------------------
    system_initstate = parameters["sys_ini"]
    system_length = 1
    eps = parameters["sys_en"]

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
    Ω = parameters["asympt_frequency"]
    K = parameters["asympt_coupling"]

    α = readdlm(parameters["MC_alphas"]) # α[l,1] = Re(αₗ) and α[l,2] = Im(αₗ)...
    β = readdlm(parameters["MC_betas"])
    w = readdlm(parameters["MC_coups"])

    mcω = @. Ω - 2K * α[:, 2]
    mcγ = @. -4K * α[:, 1]
    mcg = @. -2K * β[:, 2]
    mcζ = @. K * (w[:, 1] + im * w[:, 2])
    closure_length = length(mcω)

    sites = siteinds("vS=1/2", system_length + chain_length + closure_length)
    initialstate = MPS(
        sites, [system_initstate; repeat(["Dn"], chain_length + closure_length)]
    )

    targetop = MPS(sites, ["vN"; repeat(["vId"], length(sites) - 1)])
    opgrade = "even"
    # Unitary part of master equation
    # -------------------------------
    # -i [H, ρ] = -i H ρ + i ρ H
    adjℓ = OpSum()

    # System Hamiltonian
    # (We assume system_length == 1 for now...)
    adjℓ += -eps * gkslcommutator("N", 1)

    if chain_length > 0
        # System-chain interaction:
        if lowercase(parameters["interaction_type"]) == "xx"
            adjℓ += coups[1] * gkslcommutator("σy", 1, "σx", 2)
        elseif lowercase(parameters["interaction_type"]) == "exchange"
            adjℓ += -coups[1] * gkslcommutator("σ+", 1, "σ-", 2)
            adjℓ += -coups[1] * gkslcommutator("σ-", 1, "σ+", 2)
        else
            throw(
                error("Unrecognized interaction type. Please use \"xx\" or \"exchange\".")
            )
        end

        # Hamiltonian of the chain stub:
        # - local frequency terms
        for j in 1:chain_length
            adjℓ += -freqs[j] * gkslcommutator("N", system_length + j)
        end
        # - coupling between sites
        for j in 1:(chain_length - 1)
            # coups[1] is the coupling coefficient between the open system and the first
            # site of the chain; we don't need it here.
            site1 = system_length + j
            site2 = system_length + j + 1
            adjℓ += -coups[j + 1] * gkslcommutator("σ+", site1, "σ-", site2)
            adjℓ += -coups[j + 1] * gkslcommutator("σ-", site1, "σ+", site2)
        end
    end

    # Hamiltonian of the closure:
    # - local frequency terms
    for k in 1:closure_length
        pmsite = system_length + chain_length + k
        adjℓ += -mcω[k] * gkslcommutator("N", pmsite)
    end
    # - coupling between pseudomodes
    for k in 1:(closure_length - 1)
        pmode_site1 = system_length + chain_length + k
        pmode_site2 = system_length + chain_length + k + 1
        adjℓ += -mcg[k] * gkslcommutator("σ-", pmode_site1, "σ+", pmode_site2)
        adjℓ += -mcg[k] * gkslcommutator("σ+", pmode_site1, "σ-", pmode_site2)
    end
    # - coupling between the end of the chain stub and each pseudomode
    for j in 1:closure_length
        # Here come the Pauli strings...
        chainedge_site = system_length + chain_length
        pmode_site = system_length + chain_length + j
        ps_length = pmode_site - chainedge_site - 1 # == j-1

        paulistring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        adjℓ +=
            -(-1)^ps_length *
            mcζ[j] *
            gkslcommutator(zip(paulistring, chainedge_site:pmode_site)...)
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

        paulistring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        adjℓ +=
            -(-1)^ps_length *
            conj(mcζ[j]) *
            gkslcommutator(zip(paulistring, chainedge_site:pmode_site)...)
    end

    # Dissipative part of the master equation
    # ---------------------------------------
    gradefactor = opgrade == "even" ? 1 : -1
    for j in 1:closure_length
        pmode_site = system_length + chain_length + j
        # a ρ a†
        opstring = [repeat(["σz⋅ * ⋅σz"], pmode_site - 1); "σ+⋅ * ⋅σ-"]
        adjℓ += (
            gradefactor * mcγ[j], collect(Iterators.flatten(zip(opstring, 1:pmode_site)))...
        )
        # -0.5 (a† a ρ + ρ a† a)
        adjℓ += -0.5mcγ[j], "N⋅", pmode_site
        adjℓ += -0.5mcγ[j], "⋅N", pmode_site
    end
    L = MPO(adjℓ, sites)

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    adaptiveadjtdvp1vec!(
        targetop,
        initialstate,
        L,
        timestep,
        tmax,
        parameters["ms_stride"] * parameters["tstep"],
        sites;
        hermitian=false,
        normalize=false,
        progress=true,
        exp_tol=parameters["exp_tol"],
        krylovdim=parameters["krylov_dim"],
        io_file=parameters["out_file"],
        io_ranks=parameters["ranks_file"],
        io_times=parameters["times_file"],
        convergence_factor_bonddims=parameters["convergence_factor_bondadapt"],
        max_bond=parameters["max_bond"]
    )
end
