using PseudomodesTTEDOPA, ITensors

"""
    space(::SiteType"FDot3")

Create the Hilbert space for a site of type "FDot3".

No conserved symmetries and quantum number labels are provided for this space.
"""
function ITensors.space(::SiteType"FDot3")
    return 2^3
end

occ = [
    0
    1
]
emp = [
    1
    0
]

up = [
    0 0
    1 0
]
dn = [
    0 1
    0 0
]
id = [
    1 0
    0 1
]
fs = [
    1 0
    0 -1
] # fermion string operator

# basis order:
# e_1 -> |∅⟩
# e_2 -> c₁†|∅⟩
# e_3 -> c₂†|∅⟩
# e_4 -> c₁† c₂†|∅⟩
# e_5 -> c₃†|∅⟩
# e_6 -> c₁† c₃†|∅⟩
# e_7 -> c₂† c₃†|∅⟩
# e_8 -> c₁† c₂† c₃†|∅⟩
ITensors.state(::StateName"Emp", ::SiteType"FDot3") = kron(emp, emp, emp)
ITensors.state(::StateName"0", st::SiteType"FDot3") = state(StateName("Emp"), st)
ITensors.state(::StateName"Vac", st::SiteType"FDot3") = state(StateName("Emp"), st)
ITensors.state(::StateName"Vacuum", st::SiteType"FDot3") = state(StateName("Emp"), st)

#                                                              Modes:
#                                                          (3)  (2)  (1)
ITensors.state(::StateName"1", ::SiteType"FDot3") = kron(emp, emp, occ)
ITensors.state(::StateName"2", ::SiteType"FDot3") = kron(emp, occ, emp)
ITensors.state(::StateName"12", ::SiteType"FDot3") = kron(emp, occ, occ)
ITensors.state(::StateName"3", ::SiteType"FDot3") = kron(emp, emp, occ)
ITensors.state(::StateName"13", ::SiteType"FDot3") = kron(occ, emp, occ)
ITensors.state(::StateName"23", ::SiteType"FDot3") = kron(occ, occ, emp)
ITensors.state(::StateName"123", ::SiteType"FDot3") = kron(occ, occ, occ)

function ITensors.op(::OpName"Id", ::SiteType"FDot3")
    return Matrix(1.0I, 2^3, 2^3)
end

function ITensors.op(::OpName"c1", ::SiteType"FDot3")
    return kron(id, id, dn)
end
function ITensors.op(::OpName"c2", ::SiteType"FDot3")
    return kron(id, dn, fs)
end
function ITensors.op(::OpName"c3", ::SiteType"FDot3")
    return kron(dn, fs, fs)
end

function ITensors.op(::OpName"c†1", ::SiteType"FDot3")
    return kron(id, id, up)
end
function ITensors.op(::OpName"c†2", ::SiteType"FDot3")
    return kron(id, up, fs)
end
function ITensors.op(::OpName"c†3", ::SiteType"FDot3")
    return kron(up, fs, fs)
end

function ITensors.op(::OpName"n1", st::SiteType"FDot3")
    return ITensors.op(OpName("c†1"), st) * ITensors.op(OpName("c1"), st)
end
function ITensors.op(::OpName"n2", st::SiteType"FDot3")
    return ITensors.op(OpName("c†2"), st) * ITensors.op(OpName("c2"), st)
end
function ITensors.op(::OpName"n3", st::SiteType"FDot3")
    return ITensors.op(OpName("c†3"), st) * ITensors.op(OpName("c3"), st)
end
function ITensors.op(::OpName"ntot", st::SiteType"FDot3")
    return ITensors.op(OpName("n1"), st) +
           ITensors.op(OpName("n2"), st) +
           ITensors.op(OpName("n3"), st)
end
function ITensors.op(::OpName"ntot^2", st::SiteType"FDot3")
    return (ITensors.op(OpName("ntot"), st))^2
end

function ITensors.op(::OpName"F1", st::SiteType"FDot3")
    return kron(id, id, fs)
end
function ITensors.op(::OpName"F2", st::SiteType"FDot3")
    return kron(id, fs, id)
end
function ITensors.op(::OpName"F3", st::SiteType"FDot3")
    return kron(fs, id, id)
end
function ITensors.op(::OpName"F", st::SiteType"FDot3")
    return kron(fs, fs, fs)
end

function dot_hamiltonian(
    ::SiteType"FDot3", energies, coulomb_repulsion, sitenumber::Int
)
    E = OpSum()
    for k in 1:3
        E += (energies[k], "n$k", sitenumber)
    end

    N = OpSum()
    N += ("ntot", sitenumber)

    return E + 0.5coulomb_repulsion * (Ops.expand(N^2) - N)
end

function exchange_interaction(::SiteType"FDot3", s1::Index, s2::Index)
    stypes1 = sitetypes(s1)
    stypes2 = sitetypes(s2)
    if (SiteType("FDot3") in stypes1) && !(SiteType("FDot3") in stypes2)
        return exchange_interaction(st, sitenumber(s1), sitenumber(s2))
    elseif (SiteType("FDot3") in stypes2) && !(SiteType("FDot3") in stypes1)
        return exchange_interaction(st, sitenumber(s2), sitenumber(s1))
    else
        # Return an error if no implementation is found for any type.
        return throw(
            ArgumentError("No FDot3 site type found in either $(tags(s1)) or $(tags(s2)).")
        )
    end
end

function exchange_interaction(::SiteType"FDot3", dot_site::Int, other_site::Int)
    h = OpSum()

    for k in 1:3
        h += "c†$k", dot_site, "c", other_site
        h += "c†", other_site, "c$k", dot_site
    end
    # Doesn't work... we can't mix "c" operators from the Fermion SiteType with other
    # SiteType operators.
    return h
end
