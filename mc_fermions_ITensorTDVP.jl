using ITensors
using ITensorTDVP, Observers, Printf
using KrylovKit # for `exponentiate`
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

    # Input: chain stub parameters
    # ----------------------------
    tedopa_coefficients = readdlm(
        parameters["tedopa_coefficients"], ',', Float64; skipstart=1
    )
    coups = tedopa_coefficients[:, 1]
    freqs = tedopa_coefficients[:, 2]
    chain_length = parameters["chain_length"]
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
    vecρ = MPS(sites, [system_initstate; repeat(["Dn"], chain_length + closure_length)])

    # Unitary part of master equation
    # -------------------------------
    # -i [H, ρ] = -i H ρ + i ρ H
    ℓ = OpSum()

    # System Hamiltonian
    # (We assume system_length == 1 for now...)
    ℓ += eps * gkslcommutator("N", 1)

    if chain_length > 0
        # System-chain interaction:
        if lowercase(parameters["interaction_type"]) == "xx"
            ℓ += -coups[1] * gkslcommutator("σy", 1, "σx", 2)
        elseif lowercase(parameters["interaction_type"]) == "exchange"
            ℓ += coups[1] * gkslcommutator("σ+", 1, "σ-", 2)
            ℓ += coups[1] * gkslcommutator("σ-", 1, "σ+", 2)
        else
            throw(
                error("Unrecognized interaction type. Please use \"xx\" or \"exchange\".")
            )
        end

        # Hamiltonian of the chain stub:
        # - local frequency terms
        for j in 1:chain_length
            ℓ += freqs[j] * gkslcommutator("N", system_length + j)
        end
        # - coupling between sites
        for j in 1:(chain_length - 1)
            # coups[1] is the coupling coefficient between the open system and the first
            # site of the chain; we don't need it here.
            site1 = system_length + j
            site2 = system_length + j + 1
            ℓ += coups[j + 1] * gkslcommutator("σ+", site1, "σ-", site2)
            ℓ += coups[j + 1] * gkslcommutator("σ-", site1, "σ+", site2)
        end
    end

    # Hamiltonian of the closure:
    # - local frequency terms
    for k in 1:closure_length
        pmsite = system_length + chain_length + k
        ℓ += mcω[k] * gkslcommutator("N", pmsite)
    end
    # - coupling between pseudomodes
    for k in 1:(closure_length - 1)
        pmode_site1 = system_length + chain_length + k
        pmode_site2 = system_length + chain_length + k + 1
        ℓ += mcg[k] * gkslcommutator("σ-", pmode_site1, "σ+", pmode_site2)
        ℓ += mcg[k] * gkslcommutator("σ+", pmode_site1, "σ-", pmode_site2)
    end
    # - coupling between the end of the chain stub and each pseudomode
    for j in 1:closure_length
        # Here come the Pauli strings...
        chainedge_site = system_length + chain_length
        pmode_site = system_length + chain_length + j
        ps_length = pmode_site - chainedge_site - 1 # == j-1

        paulistring = ["σ+"; repeat(["σz"], ps_length); "σ-"]
        ℓ +=
            (-1)^ps_length *
            mcζ[j] *
            gkslcommutator(zip(paulistring, chainedge_site:pmode_site)...)

        paulistring = ["σ-"; repeat(["σz"], ps_length); "σ+"]
        ℓ +=
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
        ℓ += (mcγ[j], collect(Iterators.flatten(zip(opstring, 1:pmode_site)))...)
        # -0.5 (a† a ρ + ρ a† a)
        ℓ += -0.5mcγ[j], "N⋅", pmode_site
        ℓ += -0.5mcγ[j], "⋅N", pmode_site
    end
    L = MPO(ℓ, sites)

    vecρ0, overlap = stretchBondDim(vecρ, parameters["max_bond"])

    timestep = parameters["tstep"]
    tmax = parameters["tmax"]

    vecnS = MPS(sites, ["vN"; repeat(["vId"], length(sites) - 1)])
    vecI = MPS(sites, repeat(["vId"], length(sites)))
    function sysn(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return real(dot(vecnS, psi))
        end
        return nothing
    end
    function currenttime(; current_time, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return current_time
        end
        return nothing
    end
    function trace(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return real(dot(vecI, psi))
        end
        return nothing
    end

    obs = Observer("trace" => trace, "time" => currenttime, "sysn" => sysn)

    # ITensorTDVP doesn't export the `exponentiate_solver` name so we need
    # to define it here
    function exponentiate_solver(; kwargs...)
        function solver(H, t, psi0; kws...)
            solver_kwargs = (;
                ishermitian=get(kwargs, :ishermitian, true),
                issymmetric=get(kwargs, :issymmetric, true),
                tol=get(kwargs, :solver_tol, 1E-12),
                krylovdim=get(kwargs, :solver_krylovdim, 30),
                maxiter=get(kwargs, :solver_maxiter, 100),
                verbosity=get(kwargs, :solver_outputlevel, 0),
                eager=true,
            )
            psi, info = exponentiate(H, t, psi0; solver_kwargs...)
            return psi, info
        end
        return solver
    end

    stime = @elapsed begin
        tdvp(
            exponentiate_solver(;
                exp_tol=parameters["exp_tol"],
                krylovdim=parameters["krylov_dim"],
                ishermitian=false,
                issymmetric=false,
            ),
            L,
            tmax,
            vecρ0;
            time_step=timestep,
            normalize=false,
            (observer!)=obs,
            mindim=parameters["max_bond"],
            maxdim=parameters["max_bond"] + 1,
            cutoff=parameters["discarded_w"],
        )
        #mindim=parameters["MP_minimum_bond_dimension"],
        #maxdim=parameters["MP_maximum_bond_dimension"],

        # A partire dai risultati costruisco delle matrici da dare poi in pasto
        # alle funzioni per i grafici e le tabelle di output
        io_handle = open(parameters["out_file"], "w")
        @printf(io_handle, "%20s", "time")
        @printf(io_handle, "%20s", "trace")
        @printf(io_handle, "%20s", "sysn")
        @printf(io_handle, "\n")
        tout = results(obs, "time")
        statenorm = results(obs, "trace")
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
end
