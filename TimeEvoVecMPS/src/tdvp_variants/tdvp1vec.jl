export tdvp1vec!, adaptivetdvp1vec!

using ITensors: position!

"""
    tdvp1vec!(solver, ρ::MPS, L::Vector{MPO}, Δt::Number, tf::Number, sites; kwargs...)

Integrate the equation of motion ``d/dt ρₜ = Lᵢ(ρₜ)`` using the one-site TDVP algorithm,
where `ρ` represents the state of the system and the elements of `L` form… in some way… the
evolution operator.
The state `ρ` is assumed to be a density matrix in a vectorized form, so that the equation
of motion can be a more general master equation such as the GKSL one.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ρ::MPS`: the state of the system.
- `L::Vector{MPO}`: a list of MPOs.
- `Δt::Number`: time step of the evolution.
- `tf::Number`: end time of the evolution.
- `sites`: a collection of sites, on which `ρ` and `L` are defined.

# Implementation
For vectorized state it is still unclear whether the measurements can be made before
the sweep is complete. Therefore, until this question gets an answer, this function
postpones the measurements of all observables until all the sites of the state are
updated.
"""
function tdvp1vec!(
    solver, psi0::MPS, Ls::Vector{MPO}, time_step::Number, tf::Number, sites; kwargs...
)
    # (Copied from ITensorsTDVP)
    for L in Ls
        ITensors.check_hascommoninds(siteinds, L, psi0)
        ITensors.check_hascommoninds(siteinds, L, psi0')
    end
    Ls .= ITensors.permute.(Ls, Ref((linkind, siteinds, linkind)))
    PLs = ProjMPOSum(Ls)
    return tdvp1vec!(solver, psi0, PLs, time_step, tf, sites; kwargs...)
end

"""
    tdvp1vec!(solver, ρ::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)

Integrate the equation of motion ``d/dt ρₜ = L(ρₜ)`` using the one-site TDVP algorithm,
where `ρ` represents the state of the system and `L` the evolution operator.
The state `ρ` is assumed to be a density matrix in a vectorized form, so that the equation
of motion can be a more general master equation such as the GKSL one.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ρ::MPS`: the state of the system.
- `L::MPO`: the evolution operator.
- `Δt::Number`: time step of the evolution.
- `tf::Number`: end time of the evolution.
- `sites`: a collection of sites, on which `ρ` and `L` are defined.

# Implementation
For vectorized state it is still unclear whether the measurements can be made before
the sweep is complete. Therefore, until this question gets an answer, this function
postpones the measurements of all observables until all the sites of the state are
updated.
"""
function tdvp1vec!(solver, state::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)
    return tdvp1vec!(solver, state, ProjMPO(L), Δt, tf, sites; kwargs...)
end

"""
    tdvp1vec!(solver, ρ::MPS, L, Δt::Number, tf::Number, sites; kwargs...)

Integrate the equation of motion ``d/dt ρₜ = ℒ(ρₜ)`` using the one-site TDVP algorithm,
where `ρ` represents the state of the system and `L` the evolution operator.
The state `ρ` is assumed to be a density matrix in a vectorized form, so that the equation
of motion can be a more general master equation such as the GKSL one.

# Arguments
- `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`.
- `ρ::MPS`: the state of the system.
- `L`: a ProjMPO-like object encoding the evolution operator.
- `Δt::Number`: time step of the evolution.
- `tf::Number`: end time of the evolution.
- `sites`: a collection of sites, on which `ρ` and `L` are defined.

# Implementation
For vectorized state it is still unclear whether the measurements can be made before
the sweep is complete. Therefore, until this question gets an answer, this function
postpones the measurements of all observables until all the sites of the state are
updated.
"""
function tdvp1vec!(solver, state::MPS, PH, Δt::Number, tf::Number, sites; kwargs...)
    nsteps = Int(tf / Δt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t)  ==>  v(t) = exp(tL) v(0).

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration.
    orthogonalize!(state, 1)
    ITensors.set_nsite!(PH, 1)
    ITensors.position!(PH, state, 1)

    current_time = 0.0
    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    solver,
                    PH,
                    state,
                    site,
                    0.5Δt;
                    current_time=(ha == 1 ? current_time + 0.5Δt : current_time + Δt),
                    sweepdir=sweepdir,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
            end
        end
        current_time += Δt

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        for site in 1:N
            apply!(
                cb,
                state;
                t=current_time,
                bond=site,
                sweepend=true,
                sweepdir="right", # This value doesn't matter.
                alg=TDVP1(),
            )
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", current_time),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(
                    io_handle,
                    cb,
                    state;
                    psi0=state0,
                    vectorized=true,
                    sites=sites,
                    kwargs...,
                )
            else
                printoutput_data(
                    io_handle, cb, state; vectorized=true, sites=sites, kwargs...
                )
            end
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

"""
    adaptivetdvp1vec!(solver, state::MPS, H::MPO, Δt::Number, tf::Number, sites; kwargs...)

Like `tdvp1vec!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1vec!`](@ref).
"""
function adaptivetdvp1vec!(
    solver, psi0::MPS, Ls::Vector{MPO}, time_step::Number, tf::Number, sites; kwargs...
)
    # (Copied from ITensorsTDVP)
    for H in Hs
        ITensors.check_hascommoninds(siteinds, H, psi0)
        ITensors.check_hascommoninds(siteinds, H, psi0')
    end
    Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return tdvp1vec!(solver, psi0, PHs, time_step, tf, sites; kwargs...)
end

"""
    adaptivetdvp1vec!(solver, state::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)

Like `tdvp1vec!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1vec!`](@ref).
"""
function adaptivetdvp1vec!(solver, state::MPS, L::MPO, Δt::Number, tf::Number, sites; kwargs...)
    return adaptivetdvp1vec!(solver, state, ProjMPO(L), Δt, tf, sites; kwargs...)
end

"""
    adaptivetdvp1vec!(solver, state::MPS, L, Δt::Number, tf::Number, sites; kwargs...)

Like `tdvp1vec!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.

See [`tdvp1vec!`](@ref).
"""
function adaptivetdvp1vec!(solver, state::MPS, PH, Δt::Number, tf::Number, sites; kwargs...)
    nsteps = Int(tf / Δt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)
    convergence_factor_bonddims = get(kwargs, :convergence_factor_bonddims, 1e-4)
    max_bond = get(kwargs, :max_bond, maxlinkdim(state))

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t).

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    current_time = 0.0
    for s in 1:nsteps
        orthogonalize!(state, 1)
        ITensors.set_nsite!(PH, 1)
        position!(PH, state, 1)

        # Before each sweep, we grow the bond dimensions a bit.
        # See Dunnett and Chin, 2020 [arXiv:2007.13528v2].
        @debug "[Step $s] Attempting to grow the bond dimensions."
        adaptbonddimensions!(state, PH, max_bond, convergence_factor_bonddims)

        stime = @elapsed begin
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                                  solver,
                    PH,
                    state,
                    site,
                    0.5Δt;
                    current_time=(ha == 1 ? current_time + 0.5Δt : current_time + Δt),
                    sweepdir=sweepdir,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
            end
        end
        current_time += Δt

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        for site in 1:N
            apply!(
                cb,
                state;
                t=current_time,
                bond=site,
                sweepend=true,
                sweepdir="right", # This value doesn't matter.
                alg=TDVP1(),
            )
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", current_time),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(
                    io_handle,
                    cb,
                    state;
                    psi0=state0,
                    vectorized=true,
                    sites=sites,
                    kwargs...,
                )
            else
                printoutput_data(
                    io_handle, cb, state; vectorized=true, sites=sites, kwargs...
                )
            end
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
