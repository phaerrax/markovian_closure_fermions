export tdvp2!, tdvpMC!, tdvp1!, tdvp1vec!

using ITensors: position!

zerosite!(PH::ProjMPO) = (PH.nsite = 0)
singlesite!(PH::ProjMPO) = (PH.nsite = 1)
twosite!(PH::ProjMPO) = (PH.nsite = 2)

struct TDVP1 end
struct TDVP2 end

"""
    tdvp2!(ψ, H::MPO, timestep, endtime; kwargs...)
Evolve the MPS `ψ` up to time `endtime` using the two-site time-dependent variational
principle as described in [1].

# Keyword arguments:
All keyword arguments controlling truncation which are accepted by ITensors.replacebond!,
namely:
- `maxdim::Int`: If specified, keep only `maxdim` largest singular values after
applying the gate.
- `mindim::Int`: Minimal number of singular values to keep if truncation is performed
according to value specified by `cutoff`.
- `cutoff::Float`: If specified, keep the minimal number of singular values such that the
discarded weight is smaller than `cutoff` (but the bond dimension will be kept smaller
than `maxdim`).
- `absoluteCutoff::Bool`: If `true` truncate all singular-values whose square is smaller
than `cutoff`.

In addition the following keyword arguments are supported:
- `hermitian::Bool` (`true`) : whether the MPO `H` represents an Hermitian operator.
This will be passed to the Krylov exponentiation routine (`KrylovKit.exponentiate`) which
will in turn use a Lancosz algorithm in the case of an hermitian operator.
- `exp_tol::Float` (1e-14) : The error tolerance for `KrylovKit.exponentiate` (note that
default value was not optimized yet, so you might want to play around with it).
- `progress::Bool` (`true`) : If `true` a progress bar will be displayed

# References:
[1] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
“Unifying time evolution and optimization with matrix product states”
Physical Review B, 94(16).
https://doi.org/10.1103/PhysRevB.94.165116
"""
function tdvp2!(ψ, H::MPO, timestep, endtime; kwargs...)
    nsteps = Int(endtime / timestep)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)

    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    Δt = im * timestep
    # If `timestep` is imaginary and imag(timestep) > 0, this gives us an evolution operator
    # of the form U(Δt) = exp(-Δt H) which denotes an "imaginary-time" evolution.
    imag(Δt) == 0 && (Δt = real(Δt))
    # A unitary evolution is associated to a real `timestep`. In this case, Δt is purely
    # imaginary, as it should.
    # Otherwise, with an imaginary-time evolution, Δt is real, but the Type of the variable
    # is Complex, so we truncate any imaginary part away.
    # (`real` doesn't just chop off the imaginary part, it also converts the type from
    # Complex{T} to T.)

    store_psi0 = get(kwargs, :store_psi0, false)
    store_psi0 && (psi0 = copy(ψ))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(ψ)
    orthogonalize!(ψ, 1)
    # Move the orthogonality centre to site 1, i.e. make ψ right-canonical.
    PH = ProjMPO(H)
    position!(PH, ψ, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (b, ha) in sweepnext(N)
                # 1. Evolve with two-site Hamiltonian.
                #    ---------------------------------
                #    We project the Hamiltonian on the current bond b and the next one,
                #    then we evolve the (b, b+1) block for half a time step.
                #    The sweepnext iterator takes care of the correct indices: the pair
                #    (b, ha) here takes the values
                #       (1, 1), …, (N-1, 1), (N-1, 2), …, (1, 2)
                #    so that the correct pair of bonds is always (b, b+1).
                twosite!(PH)
                ITensors.position!(PH, ψ, b)
                wf = ψ[b] * ψ[b + 1]
                wf, info = exponentiate(
                    PH, -0.5Δt, wf; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim
                )

                info.converged == 0 && throw("exponentiate did not converge")
                # Replace the ITensors of the MPS `ψ` at sites b and b+1 with `wf`,
                # which is factorized according to the orthogonalization specification
                # given by `ortho` (left for a left-to-right sweep, right otherwise).
                # (replacebond! normalizes the result, since we pass :normalize="true"
                # within the kwargs).
                spec = replacebond!(
                    ψ,
                    b,
                    wf;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # spec is the spectrum (aka the singular values?) of the SVD.
                # Some types of callback objects might need it later, in order to compute
                # the entropy or other related quantities.

                # 2. Measure the observables.
                #    ------------------------
                #    When we are sweeping right-to-left, once the block at sites (b, b+1)
                #    has been evolved, the tensor at ψ[b + 1] has completed its evolution
                #    within the time step dt.
                #    The MPS is
                #    • left-orthogonal from ψ[1] to ψ[b - 1]
                #    • right-orthogonal from ψ[b] to ψ[end]
                #    so this is a good time to measure observables that are local to
                #    site b+1: when contracting in inner(ψ', A(n), ψ), all the sites
                #    left of ψ[b] (excluded) give the identity, and so do all those
                #    right of ψ[b + 1].
                #    The measurement can then be performed using the tensor composed by
                #    only ψ[b] and ψ[b + 1].
                apply!(
                    cb,
                    ψ;
                    t=s * timestep,
                    # This is only for storage purposes; we need the original `timestep`.
                    bond=b,
                    sweepend=(ha == 2), # apply! is skipped if ha == 1
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                    alg=TDVP2(),
                )

                # 3. Evolve with single-site Hamiltonian backward in time.
                #    -----------------------------------------------------
                #    Evolve the "next" block backwards for half a time step.
                #    Which block is the "next" block depends on the direction of the sweep:
                #    • when sweeping left-to-right, the pivot is on site `b`, and the next
                #      tensor is the one to the right, so `b+1`;
                #    • when sweeping right-to-left, the pivot is on site `b+1`, and the
                #      next tensor is the one to the left, so `b`.
                #    This step is not necessary in the case of imaginary time-evolution [1].
                i = ha == 1 ? b + 1 : b
                if 1 < i < N && !(dt isa Complex)
                    singlesite!(PH)
                    ITensors.position!(PH, ψ, i)
                    ψ[i], info = exponentiate(
                        PH,
                        0.5Δt,
                        ψ[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # dt isa Complex <==> imaginary-time evolution.
                    # TODO not sure if this is necessary anymore
                    ψ[i] /= sqrt(real(scalar(dag(ψ[i]) * ψ[i])))
                end
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", timestep * s),
                ("dt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(ψ)),
            ],
        )

        if !isempty(measurement_ts(cb)) && timestep * s ≈ measurement_ts(cb)[end]
            if store_psi0
                printoutput_data(io_handle, cb, ψ; psi0=psi0, kwargs...)
            else
                printoutput_data(io_handle, cb, ψ; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, ψ)
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
    tdvpMC!(state, H::MPO, dt, tf; kwargs...)

Evolve the MPS `state` using the MPO `H` from 0 to `tf` using an integration step `dt`.
"""
function tdvpMC!(state, H::MPO, dt, tf; kwargs...)
    nsteps = Int(tf / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, false) # Lindblad superoperator is not Hermitian
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false) # Vectorized states don't need normalization
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    #Preamble
    #println(BLAS.get_config());
    #print("NumThreads: ");

    #println(BLAS.get_num_threads())
    #flush(stdout)

    τ = im * dt
    store_initstate = get(kwargs, :store_psi0, false)
    imag(τ) == 0 && (τ = real(τ))

    # Copy the initial state if store_initstate is true
    store_initstate && (initstate = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (bond, ha) in sweepnext(N)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                #
                # 1st step
                # --------
                # Evolve using two-site Hamiltonian: we extract the tensor on sites (j,j+1)
                # and we solve locally the equations of motion, then we put the result in
                # place of the original tensor.
                #
                # We set PH.nsite to 2, since we want to use a two-site update method,
                # and we shift the projection PH of H such that the set of unprojected
                # sites begins at site bond.
                twosite!(PH)
                ITensors.position!(PH, state, bond)

                # Completing a single left-to-right sweep is a first-order integrator that
                # produces an updated |ψ⟩ at time t + τ with a local integration error of
                # order O(τ²). Completing the right-to-left sweep is equivalent to
                # composing this integrator with its adjoint, resulting in a second-order
                # symmetric method so that the state at time t + 2τ has a more favourable
                # error of order O(τ³). It is thus natural to set τ → τ/2 and to define
                # the complete sweep (left and right) as a single integration step.
                # [`Unifying time evolution and optimization with matrix product states'.
                # Jutho Haegeman, Christian Lubich, Ivan Oseledets, Bart Vandereycken and
                # Frank Verstraete, Physical Review B 94, 165116, p. 10 October 2016]
                twositeblock = state[bond] * state[bond + 1]
                twositeblock, info = exponentiate(
                    PH,
                    -0.5τ,
                    twositeblock;
                    ishermitian=hermitian,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                )
                info.converged == 0 && throw("exponentiate did not converge")

                # Factorize the twositeblock tensor in two, and replace the ITensors at
                # sites bond and bond + 1 inside the MPS state.
                spec = replacebond!(
                    state,
                    bond,
                    twositeblock;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # normalize && ( state[dir=="left" ? bond+1 : bond] /= sqrt(sum(eigs(spec))) )

                apply!(
                    cb,
                    state;
                    t=s * dt,
                    bond=bond,
                    sweepend=(ha == 2),
                    # apply! does nothing if sweepend is false, so this way we are doing
                    # the measurement only on the second sweep, from right to left.
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                    alg=TDVP2(),
                )

                # 2nd step
                # --------
                # Evolve the second site of the previous two using the single-site
                # Hamiltonian, backward in time.
                # In the case of imaginary time-evolution this step is not necessary (see
                # Ref. 1).
                i = (ha == 1 ? bond + 1 : bond)
                # The "second site" is the one in the direction of the sweep.
                if 1 < i < N && !(dt isa Complex)
                    singlesite!(PH)
                    ITensors.position!(PH, state, i)
                    state[i], info = exponentiate(
                        PH,
                        0.5τ,
                        state[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # Normalization is not necessary (or even wrong...) if the state
                    # is a density matrix.
                    #state[i] /= sqrt(real(scalar(dag(state[i]) * state[i])))
                end
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", dt * s),
                ("dt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && dt * s ≈ measurement_ts(cb)[end]
            if store_initstate
                printoutput_data(io_handle, cb, state; psi0=initstate, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
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

function tdvp1!(state, H::MPO, timestep, tf; kwargs...)
    nsteps = Int(tf / timestep)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Usually TDVP is used for ordinary time evolution according to a Hamiltonian given
    # by `H`, and a real-valued time step `timestep`, combined in the evolution operator
    # U(-itH).
    # Passing an imaginary time step iτ (and `tf`) as an argument results in an evolution
    # according to the operator U(-τH), useful for thermalization processes.
    Δt = im * timestep
    imag(Δt) == 0 && (Δt = real(Δt)) # Discard imaginary part if time step is real.

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Prepare for first iteration
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    ITensors.set_nsite!(PH, 1)
    ITensors.position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.

                # ha == 1  =>  left-to-right sweep
                # ha == 2  =>  right-to-left sweep
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    PH,
                    state,
                    site,
                    -0.5Δt; # forward by -im*timestep/2, backwards by im*timestep/2.
                    sweepdir=sweepdir,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
                apply!(
                    cb,
                    state;
                    t=s * timestep,
                    bond=site,
                    sweepend=(ha == 2),
                    sweepdir=sweepdir,
                    alg=TDVP1(),
                )
            end
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", timestep * s),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && timestep * s ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
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
    tdvp1vec!(state, H::MPO, Δt, tf, sites; kwargs...)

For vectorized state it is still unclear whether the measurements can be made before
the sweep is complete. Therefore, until this question gets an answer, this function
postpones the measurements of all observables until all the sites of the state are
updated.
"""
function tdvp1vec!(state, H::MPO, Δt, tf, sites; kwargs...)
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
    PH = ProjMPO(H)
    singlesite!(PH)
    position!(PH, state, 1)

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
                    PH,
                    state,
                    site,
                    0.5Δt;
                    sweepdir=sweepdir,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
            end
        end

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        for site in 1:N
            apply!(
                cb,
                state;
                t=Δt * s,
                bond=site,
                sweepend=true,
                sweepdir="right", # This value doesn't matter.
                alg=TDVP1(),
            )
        end

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", Δt * s),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(state)),
            ],
        )

        if !isempty(measurement_ts(cb)) && Δt * s ≈ measurement_ts(cb)[end]
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
    writeheaders_data(io_file, cb; kwargs...)

Prepare the output file `io_file`, writing the column headers for storing the data of
the observables defined in `cb`, the time steps, and other basic quantities.
"""
function writeheaders_data(io_file, cb; kwargs...)
    io_handle = nothing
    if !isnothing(io_file)
        io_handle = open(io_file, "w")
        @printf(io_handle, "%20s", "time")
        res = measurements(cb)
        for o in sort(collect(keys(res)))
            @printf(io_handle, "%40s", o)
        end
        if get(kwargs, :store_psi0, false)
            @printf(io_handle, "%40s%40s", "re_over", "im_over")
        end
        @printf(io_handle, "%40s", "Norm")
        @printf(io_handle, "\n")
    end

    return io_handle
end

"""
    writeheaders_ranks(ranks_file, N)

Prepare the output file `ranks_file`, writing the column headers for storing the data
relative to the ranks of a MPS of the given length `N`.
"""
function writeheaders_ranks(ranks_file, N)
    ranks_handle = nothing
    if !isnothing(ranks_file)
        ranks_handle = open(ranks_file, "w")
        @printf(ranks_handle, "%20s", "time")
        for r in 1:(N - 1)
            @printf(ranks_handle, "%10d", r)
        end
        @printf(ranks_handle, "\n")
    end

    return ranks_handle
end

"""
    writeheaders_stime(times_file)

Prepare the output file `times_file`, writing the column headers for the simulation
time data.
"""
function writeheaders_stime(times_file)
    times_handle = nothing
    if !isnothing(times_file)
        times_handle = open(times_file, "w")
        @printf(times_handle, "%20s", "walltime (sec)")
        @printf(times_handle, "\n")
    end

    return times_handle
end

function printoutput_data(io_handle, cb, psi; kwargs...)
    if !isnothing(io_handle)
        results = measurements(cb)
        @printf(io_handle, "%40.15f", measurement_ts(cb)[end])
        for o in sort(collect(keys(results)))
            @printf(io_handle, "%40.15f", results[o][end][1])
        end

        if get(kwargs, :store_psi0, false)
            psi0 = get(kwargs, :psi0, nothing)
            overlap = dot(psi0, psi)
            @printf(io_handle, "%40.15f%40.15f", real(overlap), imag(overlap))
        end

        # Print the norm of the trace of the state, depending on whether the MPS represents
        # a pure state or a vectorized density matrix.
        isvectorized = get(kwargs, :vectorized, false)
        if isvectorized
            @printf(io_handle, "%40.15f", real(inner(MPS(kwargs[:sites], "vecId"), psi)))
        else
            @printf(io_handle, "%40.15f", norm(psi))
        end
        @printf(io_handle, "\n")
        flush(io_handle)
    end

    return nothing
end

function printoutput_ranks(ranks_handle, cb, state)
    if !isnothing(ranks_handle)
        @printf(ranks_handle, "%40.15f", measurement_ts(cb)[end])

        for bonddim in ITensors.linkdims(state)
            @printf(ranks_handle, "%10d", bonddim)
        end

        @printf(ranks_handle, "\n")
        flush(ranks_handle)
    end

    return nothing
end

function printoutput_stime(times_handle, stime::Real)
    if !isnothing(times_handle)
        @printf(times_handle, "%20.4f\n", stime)
        flush(times_handle)
    end

    return nothing
end
