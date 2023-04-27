export adjtdvp1vec!

using ITensors: position!

"""
    adjtdvp1vec!(operator::MPS, initialstate::MPS, H::MPO, Δt, tf, meas_stride, sites; kwargs...)

Compute the time evolution, generated by the GKSL superoperator encoded in `H`, of the
operator `operator` in the Heisenberg picture, periodically measuring its expectation
value on the state `initialstate`.

# Arguments
- `operator::MPS`: an MPS representing the initial value of the operator.
- `initialstate::MPS` an MPS representing the initial state of the system.
- `H::MPO`: the MPO representing the GKSL "superoperator".
- `Δt`: the time step for the evolution.
- `tf`: the end time of the simulation.
- `meas_stride`: time between each measurement.
- `sites`: an array of ITensor sites on which the MPSs and MPOs above are defined.

# Keyword arguments
- `io_file`: output file of the measurements.
- `ranks_file`: output file for the bond dimensions of the operator's MPS.
- `times_file`: output file for the simulation time.

# Other keyword options, passed to `KrylovKit.exponentiate`
- `exp_tol::Real` -> `tol`
- `krylovdim::Int`
- `maxiter::Int`
"""
function adjtdvp1vec!(
    operator::MPS, initialstate::MPS, H::MPO, Δt, tf, meas_stride, sites; kwargs...
)
    nsteps = Int(tf / Δt)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving operator... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t).

    io_handle = open(io_file, "w")
    @printf(io_handle, "%20s", "time")
    @printf(io_handle, "%20s", "exp_val")
    @printf(io_handle, "\n")

    ranks_handle = writeheaders_ranks(ranks_file, length(operator))
    times_handle = writeheaders_stime(times_file)

    N = length(operator)

    # Prepare for first iteration.
    orthogonalize!(operator, 1)
    PH = ProjMPO(H)
    singlesite!(PH)
    position!(PH, operator, 1)

    prev_t = zero(Δt)
    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the operator MPS, not its bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.

                # The algorithm starts from a right-canonical MPS A, where each matrix
                # A(n) = Aᵣ(n) is right-orthogonal.

                # 1. Project the Hamiltonian on the current site.
                #    --------------------------------------------

                singlesite!(PH)
                ITensors.position!(PH, operator, site)

                # 2. Evolve A(site) for half the time-step Δt.
                #    -----------------------------------------

                φ, info = exponentiate(
                    PH,
                    0.5Δt,
                    operator[site];
                    ishermitian=false,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                    eager=true,
                )
                info.converged == 0 && throw("exponentiate did not converge")

                # Now we take different steps depending on whether we are at
                # the end of the half-sweep or not.
                if (ha == 1 && site != N) || (ha == 2 && site != 1)
                    # 3. Factorize the updated A(site) as Aₗ(site)C(site) such that the
                    #    matrix Aₗ is left-orthogonal.
                    #    --------------------------------------------------------------

                    Δs = (ha == 1 ? 1 : -1)
                    # site + Δs is the physical index of the next site in the sweep.

                    # Perform the SVD decomposition. Note that the group of indices
                    # provided by the second argument is interpreted as the "left index"
                    # of φ, therefore there is no need to "reverse" the indices when we
                    # are performing the right-to-left sweep: everything is taken care of
                    # by ITensors accordingly.
                    U, S, V = svd(φ, uniqueinds(φ, operator[site + Δs]))

                    operator[site] = U # This is left(right)-orthogonal if ha==1(2).
                    C = S * V
                    if ha == 1
                        ITensors.setleftlim!(operator, site)
                        # This has something to do with the range within the MPS where the
                        # orthogonality properties hold...
                    elseif ha == 2
                        ITensors.setrightlim!(operator, site)
                    end

                    # 4. Evolve C(site) backwards in time of a half-step Δt/2 and
                    #    incorporate in the matrix Aᵣ(site+1) of the next site along
                    #    the sweep.
                    #    -----------------------------------------------------------

                    # Calculate the new zero-site projection of the evolution operator.
                    zerosite!(PH)
                    position!(PH, operator, ha == 1 ? site + 1 : site)
                    # Shouldn't we have ha == 1 ? site+1 : site-1 ?

                    C, info = exponentiate(
                        PH,
                        -0.5Δt,
                        C;
                        ishermitian=false,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                        eager=true,
                    )

                    # Incorporate the backwards-evolved C(site) with the matrix on the
                    # next site.
                    operator[site + Δs] = C * operator[site + Δs]

                    if ha == 1
                        ITensors.setrightlim!(operator, site + Δs + 1)
                    elseif ha == 2
                        ITensors.setleftlim!(operator, site + Δs - 1)
                    end

                    # Reset the single-site projection of the evolution operator,
                    # ready for the next sweep.
                    singlesite!(PH)
                else
                    # There's nothing to do if the half-sweep is at the last site.
                    operator[site] = φ
                end
            end
        end
        prev_t += Δt
        t = Δt * s

        !isnothing(pbar) && ProgressMeter.next!(
            pbar;
            showvalues=[
                ("t", t),
                ("Δt step time", round(stime; digits=3)),
                ("Max bond-dim", maxlinkdim(operator)),
            ],
        )

        # Now the backwards sweep has ended, so the whole MPS of the operator is up-to-date.
        # We can then calculate the expectation values on the initial state.
        #if t - prev_t ≈ meas_stride... how does this work?
        if true
            @printf(io_handle, "%40.15f", t)
            @printf(io_handle, "%40.15f", real(inner(initialstate, operator)))
            @printf(io_handle, "\n")
            flush(io_handle)

            @printf(ranks_handle, "%40.15f", t)
            for bonddim in ITensors.linkdims(operator)
                @printf(ranks_handle, "%10d", bonddim)
            end
            @printf(ranks_handle, "\n")
            flush(ranks_handle)

            printoutput_stime(times_handle, stime)
        end
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
