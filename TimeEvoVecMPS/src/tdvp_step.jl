function exponentiate_solver(; kwargs...)
    # Default solver that we provide if no solver is given by the user.
    function solver(H, time_step, ψ₀; kws...)
        solver_kwargs = (;
            ishermitian=get(kwargs, :ishermitian, true),
            issymmetric=get(kwargs, :issymmetric, true),
            tol=get(kwargs, :solver_tol, 1E-12),
            krylovdim=get(kwargs, :solver_krylovdim, 30),
            maxiter=get(kwargs, :solver_maxiter, 100),
            verbosity=get(kwargs, :solver_outputlevel, 0),
            eager=true,
        )
        ψₜ, info = exponentiate(H, time_step, ψ₀; solver_kwargs...)
        return ψₜ, info
    end
    return solver
end

function tdvp_solver(; kwargs...)
    # Fallback solver function if no solver is specified when calling tdvp.
    solver_backend = get(kwargs, :solver_backend, "exponentiate")
    if solver_backend == "exponentiate"
        return exponentiate_solver(; kwargs...)
    else
        error(
            "solver_backend=$solver_backend not recognized " *
            "(the only option is \"exponentiate\")",
        )
    end
end

# Fallback functions if no solver is given.
function tdvp1!(state::MPS, H::MPO, timestep::Number, tf::Number; kwargs...)
    return tdvp1!(tdvp_solver(; kwargs...), state, H, timestep, tf; kwargs...)
end
function adaptivetdvp1!(state::MPS, H::MPO, timestep::Number, tf::Number; kwargs...)
    return adaptivetdvp1!(tdvp_solver(; kwargs...), state, H, timestep, tf; kwargs...)
end

function tdvp1vec!(state::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)
    return tdvp1vec!(tdvp_solver(; kwargs...), state, L, Δt, tf, sites; kwargs...)
end
function adaptivetdvp1vec!(state::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)
    return adaptivetdvp1vec!(tdvp_solver(; kwargs...), state, L, Δt, tf, sites; kwargs...)
end

function adjtdvp1vec!(
    operator::MPS,
    initialstate::MPS,
    H::MPO,
    Δt::Number,
    tf::Number,
    meas_stride::Number,
    sites;
    kwargs...,
)
    return adjtdvp1vec!(
        tdvp_solver(; kwargs...),
        operator,
        initialstate,
        H,
        Δt,
        tf,
        meas_stride,
        sites;
        kwargs...,
    )
end
function adaptiveadjtdvp1vec!(
    operator::MPS,
    initialstate::MPS,
    H::MPO,
    Δt::Number,
    tf::Number,
    meas_stride::Number,
    sites;
    kwargs...,
)
    return adaptiveadjtdvp1vec!(
        tdvp_solver(; kwargs...),
        operator,
        initialstate,
        H,
        Δt,
        tf,
        meas_stride,
        sites;
        kwargs...,
    )
end

"""
    function tdvp_site_update!(
        solver, PH, psi::MPS, i::Int, time_step;
        sweepdir, hermitian, exp_tol, krylovdim, maxiter
    )

Update site `i` of the MPS `psi` using the 1-site TDVP algorithm with time step `time_step`.
The keyword argument `sweepdir` indicates the direction of the current sweep.
"""
function tdvp_site_update!(
    solver,
    PH,
    psi::MPS,
    site::Int,
    time_step;
    sweepdir,
    current_time,
    hermitian,
    exp_tol,
    krylovdim,
    maxiter,
)
    N = length(psi)
    ITensors.set_nsite!(PH, 1)
    ITensors.position!(PH, psi, site)

    # Forward evolution half-step.
    phi, info = solver(PH, time_step, psi[site]; current_time)
    info.converged == 0 && throw("exponentiate did not converge")

    # Backward evolution half-step.
    # (it is necessary only if we're not already at the edges of the MPS)
    if (sweepdir == "right" && (site != N)) || (sweepdir == "left" && site != 1)
        new_proj_base_site = (sweepdir == "right" ? site + 1 : site)
        # When we are sweeping right-to-left and switching from a 1-site projection to a
        # 0-site one, the right-side projection moves one site to the left, but the “base”
        # site of the ProjMPO doesn't move  ==>  new_proj_base_site = site
        # In the other sweep direction, the left-side projection moves one site to the left
        # and so does the “base” site  ==>  new_proj_base_site = site + 1

        next_site = (sweepdir == "right" ? site + 1 : site - 1)
        # This is the physical index of the next site in the sweep.

        U, S, V = svd(phi, uniqueinds(phi, psi[next_site]))
        psi[site] = U # This is left(right)-orthogonal if ha==1(2).
        C = S * V
        if sweepdir == "right"
            ITensors.setleftlim!(psi, site)
        elseif sweepdir == "left"
            ITensors.setrightlim!(psi, site)
        end

        # Prepare the zero-site projection.
        ITensors.set_nsite!(PH, 0)
        ITensors.position!(PH, psi, new_proj_base_site)

        C, info = solver(PH, -time_step, C; current_time)

        # Reunite the backwards-evolved C with the matrix on the next site.
        psi[next_site] *= C

        # Now the orthocenter is on `next_site`.
        # Set the new orthogonality limits of the MPS.
        if sweepdir == "right"
            ITensors.setrightlim!(psi, next_site + 1)
        elseif sweepdir == "left"
            ITensors.setleftlim!(psi, next_site - 1)
        else
            throw("Unrecognized sweepdir: $sweepdir")
        end

        # Reset the one-site projection… and we're done!
        ITensors.set_nsite!(PH, 1)
    else
        # There's nothing to do if the half-sweep is at the last site.
        psi[site] = phi
    end
end
