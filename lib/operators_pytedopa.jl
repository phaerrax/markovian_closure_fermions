using ITensors

############################################################################
# Questo file è una modifica di lib/operators.jl in modo che gli operatori #
# siano uguali a quelli usati nel notebook di esempio di py-tedopa.        #
############################################################################

# Costruzione della lista di operatori 2-locali
function twositeoperators(sites::Vector{Index{Int64}},
                          localcfs::Vector{<:Real},
                          interactioncfs::Vector{<:Real})
  # Restituisce la lista di termini (che dovranno essere poi esponenziati nel
  # modo consono alla simulazione) dell'Hamiltoniano o del “Lindbladiano” che
  # agiscono sulle coppie di siti adiacenti della catena.
  # L'elemento list[j] è l'operatore hⱼ,ⱼ₊₁ (o ℓⱼ,ⱼ₊₁, a seconda della notazione)
  #
  # Argomenti
  # ---------
  # · `sites::Vector{Index}`: un vettore di N elementi, contenente gli Index
  #   che rappresentano i siti del sistema;
  # · `localcfs::Vector{Index}`: un vettore di N elementi contenente i
  #   coefficienti che moltiplicano i termini locali dell'Hamiltoniano o del
  #   Lindbladiano
  # · `interactioncfs::Vector{Index}`: un vettore di N-1 elementi contenente i
  #   coefficienti che moltiplicano i termini di interazione tra siti adiacenti,
  #   con la convenzione che l'elemento j è riferito al termine hⱼ,ⱼ₊₁/ℓⱼ,ⱼ₊₁.
  list = ITensor[]
  localcfs[begin] *= 2
  localcfs[end] *= 2
  # Anziché dividere per casi il ciclo che segue distinguendo i siti ai lati
  # della catena (che non devono avere il fattore 0.5 scritto sotto), moltiplico
  # qui per 2 i rispettivi coefficienti degli operatori locali.
  for j ∈ 1:length(sites)-1
    s1 = sites[j]
    s2 = sites[j+1]
    h = 0.5localcfs[j] * Hlocal(s1) * op("Id", s2) +
        0.5localcfs[j+1] * op("Id", s1) * Hlocal(s2) +
        interactioncfs[j] * Hinteraction(s1, s2)
    push!(list, h)
  end
  return list
end

# Hamiltoniani
# ============
# Hamiltoniani locali
function Hlocal(s::Index)
  if SiteType("S=1/2") ∈ sitetypes(s)
    h = 0.5op("σz", s)
  elseif SiteType("Osc") ∈ sitetypes(s)
    h = op("N", s)
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return h
end

# Hamiltoniani di interazione
function Hinteraction(s1::Index, s2::Index)
  if SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    h = op("σ-", s1) * op("σ+", s2) + op("σ+", s1) * op("σ-", s2)
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    h = 0.5 * op("X", s1) * op("N", s2) # = (a + a†) ⊗ ¼(I₂ + σᶻ)
  elseif SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    h = 0.5 * op("N", s1) * op("X", s2) # = ¼(I₂ + σᶻ) ⊗ (a + a†)
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    h = op("a+", s1) * op("a-", s2) + op("a-", s1) * op("a+", s2)
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return h
end

# Lindblad
# ========
# Lindblad locali
function ℓlocal(s::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s)
    ℓ = 0.5im * (op("σz:Id", s) - op("Id:σz", s))
  elseif SiteType("vecOsc") ∈ sitetypes(s)
    ℓ = im * (op("N:Id", s) - op("Id:N", s))
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return ℓ
end

# Lindblad di interazione
function ℓinteraction(s1::Index, s2::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s1) &&
     SiteType("vecS=1/2") ∈ sitetypes(s2)
    ℓ = -0.5im * (op("σ-:Id", s1) * op("σ+:Id", s2) +
                  op("σ+:Id", s1) * op("σ-:Id", s2) -
                  op("Id:σ-", s1) * op("Id:σ+", s2) -
                  op("Id:σ+", s1) * op("Id:σ-", s2))
  elseif SiteType("vecOsc") ∈ sitetypes(s1) &&
         SiteType("vecS=1/2") ∈ sitetypes(s2)
    ℓ = im * (op("asum:Id", s1) * op("σx:Id", s2) -
              op("Id:asum", s1) * op("Id:σx", s2))
  elseif SiteType("vecS=1/2") ∈ sitetypes(s1) &&
         SiteType("vecOsc") ∈ sitetypes(s2)
    ℓ = im * (op("σx:Id", s1) * op("asum:Id", s2) -
              op("Id:σx", s1) * op("Id:asum", s2))
  else
    throw(DomainError((s1, s2), "SiteType non riconosciuti."))
  end
  return ℓ
end
