using ITensors, PseudomodesTTEDOPA

"""
    space(::SiteType"vvFDot3")

Create the Hilbert space for a site of type "vFDot3", i.e. a mixed state describing a
site with a fermionic three-level quantum dot.
The density matrix is represented in the generalised Gell-Mann basis, composed
of Hermitian traceless matrices together with the identity matrix.

No conserved symmetries and quantum number labels are provided for this space.
"""
function ITensors.space(::SiteType"vFDot3")
    return (2^3)^2
end

# An element A ∈ Mat(ℂ⁴) is representeb by the a vector v such that
#     vᵢ = tr(Λᵢ A),
# while a linear map L : Mat(ℂ⁴) → Mat(ℂ⁴) by the matrix ℓ such that
#     ℓᵢⱼ = tr(Λᵢ L(Λⱼ)).

# Shorthand notation:
dotop(on::AbstractString) = ITensors.op(OpName(on), SiteType("FDot3"))
function vstate(sn::AbstractString)
    v = ITensors.state(StateName(sn), SiteType("FDot3"))
    return PseudomodesTTEDOPA.vec(kron(v, v'), gellmannbasis(2^3))
end
function vop(on::AbstractString)
    return PseudomodesTTEDOPA.vec(
        ITensors.op(OpName(on), SiteType("FDot3")), gellmannbasis(2^3)
    )
end
premul(y) = PseudomodesTTEDOPA.vec(x -> y * x, gellmannbasis(2^3))
postmul(y) = PseudomodesTTEDOPA.vec(x -> x * y, gellmannbasis(2^3))

# basis order:
# e_1 -> |∅⟩
# e_2 -> c₁†|∅⟩
# e_3 -> c₂†|∅⟩
# e_4 -> c₁† c₂†|∅⟩
# e_5 -> c₃†|∅⟩
# e_6 -> c₁† c₃†|∅⟩
# e_7 -> c₂† c₃†|∅⟩
# e_8 -> c₁† c₂† c₃†|∅⟩
ITensors.state(::StateName"Emp", ::SiteType"vFDot3") = vstate("Emp")
ITensors.state(::StateName"0", st::SiteType"vFDot3") = state(StateName("Emp"), st)
ITensors.state(::StateName"Vac", st::SiteType"vFDot3") = state(StateName("Emp"), st)
ITensors.state(::StateName"Vacuum", st::SiteType"vFDot3") = state(StateName("Emp"), st)

ITensors.state(::StateName"1", ::SiteType"vFDot3") = vstate("1")
ITensors.state(::StateName"2", ::SiteType"vFDot3") = vstate("2")
ITensors.state(::StateName"12", ::SiteType"vFDot3") = vstate("12")
ITensors.state(::StateName"3", ::SiteType"vFDot3") = vstate("3")
ITensors.state(::StateName"13", ::SiteType"vFDot3") = vstate("13")
ITensors.state(::StateName"23", ::SiteType"vFDot3") = vstate("23")
ITensors.state(::StateName"123", ::SiteType"vFDot3") = vstate("123")

ITensors.state(::StateName"vId", ::SiteType"vFDot3") = vop("Id")
ITensors.state(::StateName"vecId", ::SiteType"vFDot3") = vop("Id")
ITensors.state(::StateName"vn1", ::SiteType"vFDot3") = vop("n1")
ITensors.state(::StateName"vn2", ::SiteType"vFDot3") = vop("n2")
ITensors.state(::StateName"vn3", ::SiteType"vFDot3") = vop("n3")
ITensors.state(::StateName"vntot", ::SiteType"vFDot3") = vop("ntot")

function ITensors.op(on::OpName, ::SiteType"vFDot3")
    name = strip(String(ITensors.name(on)))
    dotloc = findfirst("⋅", name)
    # This returns the position of the cdot in the operator name String.
    # It is `nothing` if no cdot is found.
    if !isnothing(dotloc)
        op1, op2 = nothing, nothing
        op1 = name[1:prevind(name, dotloc.start)]
        op2 = name[nextind(name, dotloc.start):end]
        # If the OpName `on` is written correctly, i.e. it is either "A⋅" or "⋅A" for some
        # A, then either `op1` or `op2` has to be empty (not both, not neither of them).
        if (op1 == "" && op2 == "") || (op1 != "" && op2 != "")
            throw(ArgumentError("Malformed operator name: $name"))
        end
        # name = "⋅A" -> op1 is an empty string
        # name = "A⋅" -> op2 is an empty string
        if op1 == ""
            return postmul(dotop(op2))
        elseif op2 == ""
            return premul(dotop(op1))
        else
            error("Unknown error with operator name $name")
        end
    end
end

function dot_hamiltonian(
    ::SiteType"vFDot3", energies, coulomb_repulsion, sitenumber::Int
)
    E = OpSum()
    for k in 1:3
        E += energies[k] * gkslcommutator("n$k", sitenumber)
    end

    N² = gkslcommutator("ntot^2", sitenumber)
    N = gkslcommutator("ntot", sitenumber)

    return E + 0.5coulomb_repulsion * (N² - N)
end

function exchange_interaction(::SiteType"vFDot3", s1::Index, s2::Index)
    stypes1 = sitetypes(s1)
    stypes2 = sitetypes(s2)
    if (SiteType("vFDot3") in stypes1) && !(SiteType("vFDot3") in stypes2)
        return exchange_interaction(st, sitenumber(s1), sitenumber(s2))
    elseif (SiteType("vFDot3") in stypes2) && !(SiteType("vFDot3") in stypes1)
        return exchange_interaction(st, sitenumber(s2), sitenumber(s1))
    else
        # Return an error if no implementation is found for any type.
        throw(
            ArgumentError("No vFDot3 site type found in either $(tags(s1)) or $(tags(s2)).")
        )
    end
end

function exchange_interaction(::SiteType"vFDot3", dot_site::Int, other_site::Int)
    h = OpSum()

    for k in 1:3
        h +=
            energies[k] * (
                gkslcommutator("c†$k", dot_site, "σ-", other_site) +
                gkslcommutator("c$k", dot_site, "σ+", other_site)
            )
    end
    return h
end
