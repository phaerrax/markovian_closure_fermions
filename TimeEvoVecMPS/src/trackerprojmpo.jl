export TrackerProjMPO

"""
A TrackerProjMPO computes and stores the projection of an MPO into a basis defined by an
MPS, leaving a certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling the `position!` method.

Drawing of the network represented by a TrackerProjMPO `P(H)`, showing the case
of `nsite(P)==2` and `position!(P,psi,4)` for an MPS `psi`:

```
o--o--o-      -o--o--o--o--o--o <psi|
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o--o--o--o--o--o--o--o H
|  |  |  |  |  |  |  |  |  |  |
o--o--o-      -o--o--o--o--o--o |psi>
```

Differently from a standard ProjMPO object, a TrackerProjMPO also keeps tracks of the MPS
used to compute the projections by recording the IDs of its bond indices; this way, if the
state is changed in a way that affects the bond indices, the projections are recomputed
accordingly.
"""
mutable struct TrackerProjMPO <: ITensors.AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    ids::Vector{ITensors.IDType}
    H::MPO
    LR::Vector{ITensor}
end

function TrackerProjMPO(H::MPO)
    return TrackerProjMPO(
        0,
        length(H) + 1,
        2,
        Vector{ITensors.IDType}(undef, length(H) - 1),
        H,
        Vector{ITensor}(undef, length(H)),
    )
end

function Base.copy(P::TrackerProjMPO)
    return TrackerProjMPO(P.lpos, P.rpos, P.nsite, copy(P.ids), copy(P.H), copy(P.LR))
end

function ITensors.set_nsite!(P::TrackerProjMPO, nsite)
    P.nsite = nsite
    return P
end

ids(P::TrackerProjMPO) = P.ids

"""
    position!(P::ProjMPO, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the MPO represented by the ProjMPO `P` such
that the set of unprojected sites begins with site `pos`. This operation efficiently reuses
previous projections of the MPO on sites that have already been projected. The MPS `psi`
must have compatible bond indices with the previous projected MPO tensors for this
operation to succeed.
"""
function ITensors.position!(P::TrackerProjMPO, psi::MPS, pos::Int)
    # What do we need to do?
    # 1) Check whether the MPS has changed, by comparing psi's bond indices to the already
    #    stored ones.
    # 2) Find out which bonds are "new".
    # 3) Recalculate the projections associated to those sites (and only on those sites).

    # Sometimes we have to call position! when some sites of the MPS are not linked.
    # For example, during the TDVP algorithm, after, say, ψ[1] has been evolved, we
    # factorize it as ψ[1] = Q*R, set ψ[1] = Q and evolve R back in time using the ProjMPO
    # object. When we move its position on site 2, however, the MPS is disconnected, since
    # Q and ψ[2] do not share a link Index (for now: after its backwards evolution, we
    # set ψ[2] = ψ[2]*R and the MPS is connected again).
    # If we just calculated the IDs of the link indices of the state, we'd get an error
    # since the (1,2) bond is currently `nothing`.
    # So, here's a dirty hack: we see if there's a `nothing` in the link indices, and
    # do as if it were an Index with ID = 0. Hopefully, this will not match any other
    # index ID, and this method will carry on updating the projection on such bond.
    newids = [isnothing(ind) ? zero(ITensors.IDType) : id(ind) for ind in linkinds(psi)]

    # Get the (vector) indices where the IDs don't match: these are the bonds that changed
    # since the last update. More precisely, if `n` appears in this list then the bond
    # (n, n+1) has been changed in some way, and we will need to update the projection.
    bonds_with_different_id = findall(==(false), newids .== ids(P))
    @debug "State's current bonds: $(linkinds(psi))"
    @debug "Saved bond IDs: $([Int(n % 1000) for n in ids(P)])"
    # A changed bond may signify that the MPS sites it connects have been recalculated.
    # The typical situation is when we reorthogonalize the MPS and move the orthocenter
    # from n to n+1; this means factorizing psi[n] and multplying psi[n+1] by the "factor
    # to the right", so both psi[n] and psi[n+1] have changed in the end.
    if isempty(bonds_with_different_id)
        # Same behaviour as an ordinary ProjMPO.
        # Normally, we would calculate the left projections from 1 to pos-1, and
        # right projections from pos+nsite to the end. This way, the gap starting
        # [pos, pos+nsite) remains open and gives us the free indices that will
        # be applied to the state in the TDVP evolution.
        ITensors.makeL!(P, psi, pos - 1)
        ITensors.makeR!(P, psi, pos + ITensors.nsite(P))
    else
        # If some bonds have been changed, we need to force the ProjMPO methods
        # to recompute the relevant projections.
        # At the very least, for each n in bonds_with_different_id we need to
        # recalculate the projections on sites n and n+1.
        # A (hopefully) sufficient strategy is to recompute all projections
        # from min(bonds_with_different_id) to lpos, and from rpos to
        # max(bonds_with_different_id)
        @debug "Bonds $bonds_with_different_id have a new ID"
        _remakeL!(P, psi, pos - 1, bonds_with_different_id)
        _remakeR!(P, psi, pos + ITensors.nsite(P), bonds_with_different_id)
        P.ids = newids
    end
    return P
end

function _remakeL!(
    P::ITensors.AbstractProjMPO, psi::MPS, k::Int, newbonds::Vector{Int}
)::Union{ITensor,Nothing}
    # As already said: all projection terms on sites from min(newbonds) to the
    # current positions are invalid, since they contain information on the
    # now old state, and must be recomputed.
    # [ Remember that k = pos-1 ]
    lnewbonds = filter(≤(k), newbonds)
    @debug "Bonds to be changed on the left part: $lnewbonds"
    if P.lpos ≥ k && isempty(lnewbonds)
        P.lpos = k
        return nothing
    end
    # We start from:
    if isempty(lnewbonds)
        ll = P.lpos
    else
        ll = min(minimum(lnewbonds) - 1, P.lpos)
    end
    # If min(newbonds) comes before k, then more projections must be redone.
    # Those on sites to the left of ll are still valid.
    ll = max(ll, 0) # Make sure ll is at least 0 for the generic logic below
    L = (ll ≤ 0 ? ITensors.OneITensor() : P.LR[ll])
    while ll < k
        L = L * psi[ll + 1] * P.H[ll + 1] * dag(prime(psi[ll + 1]))
        P.LR[ll + 1] = L
        ll += 1
    end
    P.lpos = k
    return L
end

function _remakeR!(
    P::ITensors.AbstractProjMPO, psi::MPS, k::Int, newbonds::Vector{Int}
)::Union{ITensor,Nothing}
    # As already said: all projection terms on sites from the current position
    # to max(newbonds) are invalid, since they contain information on the
    # now old state, and must be recomputed.
    # [ Remember that k = pos+nsite ]
    rnewbonds = filter(≥(k-ITensors.nsite(P)), newbonds)
    @debug "Bonds to be changed on the right part: $rnewbonds"
    if P.rpos ≤ k && isempty(rnewbonds)
        P.rpos = k
        return nothing
    end
    N = length(P.H)
    if isempty(rnewbonds)
        rl = P.rpos
    else
        # We need to add 2 (instead of 1 as in remakeL) since
        #   maximum(rnewbonds)
        # is the rightmost bond that was changed
        #   maximum(rnewbonds) + 1
        # is the site just to the right of the changed bond, therefore
        #   maximum(rnewbonds) + 2
        # is the site of the first (from left to right) still valid projection.
        rl = max(P.rpos, maximum(rnewbonds) + 2)
    end
    rl = min(rl, N + 1) # Make sure rl is no bigger than `N + 1` for the generic logic below
    R = (rl ≥ N + 1 ? ITensors.OneITensor() : P.LR[rl])
    while rl > k
        R = R * psi[rl - 1] * P.H[rl - 1] * dag(prime(psi[rl - 1]))
        P.LR[rl - 1] = R
        rl -= 1
    end
    P.rpos = k
    return R
end
