"""
    function tdvp_site_update!(
        PH, psi::MPS, i::Int, time_step;
        sweepdir, hermitian, exp_tol, krylovdim, maxiter
    )

Update site `i` of the MPS `psi` using the 1-site TDVP algorithm with time step `time_step`.
The keyword argument `sweepdir` indicates the direction of the current sweep.
"""
function tdvp_site_update!(
    PH, psi::MPS, site::Int, time_step; sweepdir, hermitian, exp_tol, krylovdim, maxiter
)
    ITensors.set_nsite!(PH, 1)
    ITensors.position!(PH, psi, site)

    # Forward evolution half-step.
    φ, info = exponentiate(
        PH,
        time_step,
        psi[site];
        ishermitian=hermitian,
        tol=exp_tol,
        krylovdim=krylovdim,
        maxiter=maxiter,
        eager=true,
    )
    info.converged == 0 && throw("exponentiate did not converge")

    # Backward evolution half-step.
    # (it is necessary only if we're not already at the edges of the MPS)
    if (sweepdir == "rightwards" && (site != N)) || (sweepdir == "leftwards" && site != 1)
        # ha == 1  =>  left-to-right sweep
        # ha == 2  =>  right-to-left sweep

        new_proj_base_site = (sweepdir == "rightwards" ? site + 1 : site)
        # When we are sweeping right-to-left and switching from a 1-site projection to a
        # 0-site one, the right-side projection moves one site to the left, but the “base”
        # site of the ProjMPO doesn't move  ==>  new_proj_base_site = site
        # In the other sweep direction, the left-side projection moves one site to the left
        # and so does the “base” site  ==>  new_proj_base_site = site + 1

        next_site = (sweepdir == "rightwards" ? site + 1 : site - 1)
        # This is the physical index of the next site in the sweep.

        U, S, V = svd(φ, uniqueinds(φ, psi[next_site]))
        psi[site] = U # This is left(right)-orthogonal if ha==1(2).
        C = S * V
        if ha == 1
            ITensors.setleftlim!(psi, site)
        elseif ha == 2
            ITensors.setrightlim!(psi, site)
        end

        # Prepare the zero-site projection.
        ITensors.set_nsite!(PH, 0)
        ITensors.position!(PH, psi, new_proj_base_site)

        C, info = exponentiate(
            PH,
            -time_step,
            C;
            ishermitian=hermitian,
            tol=exp_tol,
            krylovdim=krylovdim,
            maxiter=maxiter,
            eager=true,
        )

        # Reunite the backwards-evolved C with the matrix on the next site.
        psi[next_site] *= C

        # Now the orthocenter is on `next_site`.
        # Set the new orthogonality limits of the MPS.
        if sweepdir == "rightwards"
            ITensors.setrightlim!(psi, next_site + 1)
        elseif sweepdir == "leftwards"
            ITensors.setleftlim!(psi, next_site - 1)
        else
            throw("Unrecognized sweepdir: $sweepdir")
        end

        # Reset the one-site projection… and we're done!
        ITensors.set_nsite!(PH, 1)
    end
end
