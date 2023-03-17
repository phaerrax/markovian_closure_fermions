# some extensions for ITensors functionality
# the goal is to eventually contribute these upstream if found appropriate

export growbond!, bondconvergencemeasure

function findprimeinds(is::IndexSet, plevel::Int=-1)
    if plevel>=0
        return filter(x->plev(x)==plevel, is)
    else
        return filter(x->plev(x)>0, is)
    end
end

findprimeinds(A::ITensor,args...) = findprimeinds(A.inds,args...)

function isleftortho(M,i)
    i==length(M) && return true
    L = M[i]*prime(dag(M[i]), i==1 ? "Link" : commonindex(M[i],M[i+1]))
    l = linkindex(M,i)
    return norm(L-delta(l,l')) < 1E-12
end

function isrightortho(M,i)
    i==1 && return true
    R = M[i]*prime(dag(M[i]),i==length(M) ? "Link" : commonindex(M[i],M[i-1]))
    r = linkindex(M,i-1)
    return norm(R-delta(r,r')) < 1E-12
end

function reorthogonalize!(psi::MPS)
    ITensors.setleftlim!(psi,-1)
    ITensors.setrightlim!(psi,length(psi)+2)
    orthogonalize!(psi,1)
    psi[1] /= sqrt(inner(psi,psi))
end

"""
    growbond!(v::MPS, bond::Integer; increment::Integer=1)

Grow the bond dimension of the MPS `v` between sites `bond` and `bond+1` by `increment`.
Return the new bond dimension.
"""
function growbond!(v::MPS, bond::Integer; increment::Integer=1)::Integer
    bond_index = commonind(v[bond], v[bond + 1])
    current_bonddim = ITensors.dim(bond_index)
    aux = Index(current_bonddim + increment; tags=tags(bond_index))
    v[bond] = v[bond] * delta(bond_index, aux)
    v[bond + 1] = v[bond + 1] * delta(bond_index, aux)
    return current_bonddim + increment
end

"""
    bondconvergencemeasure(H::MPO, state::MPS, bond::Integer)

Return a measure of the convergence for the bond dimension on the bond (`bond`, `bond+1`)
of the MPS `state` for a time-evolution determined by `H`. See [1] for details.

# References:
[1] Dunnett, Angus J. and Chin, Alex W. (2020).
“Dynamically Evolving Bond-Dimensions within the one-site
Time-Dependent-Variational-Principle method for Matrix Product States: Towards efficient
simulation of non-equilibrium open quantum dynamics”
https://doi.org/10.48550/arXiv.2007.13528
"""
function bondconvergencemeasure(H::MPO, state::MPS, bond::Integer)::Real
    v = copy(state) # So that the re-orthogonalizing doesn't touch `state`
    PH = ProjMPO(H) # We need to create a new one each time we call the function

    PH.nsite = 1
    orthogonalize!(v, bond)
    ITensors.position!(PH, v, bond)
    H1 = PH(v[bond])

    _, S, V = svd(v[bond], uniqueinds(v[bond], v[bond + 1]))
    C = S * V
    PH.nsite = 0
    ITensors.position!(PH, v, bond + 1)
    K = PH(C)

    PH.nsite = 1
    orthogonalize!(v, bond + 1)
    #ITensors.position!(PH, v, bond+1)
    H2 = PH(v[bond + 1])

    return real(scalar(H1 * H1) + scalar(H2 * H2) + scalar(K * K))
end
