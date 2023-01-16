using ITensors
using LinearAlgebra

const ⊗ = kron

#=
In questo file definisco gli stati e gli operatori con cui può essere descritto
un oscillatore armonico, sia nella versione normale che in quella vettorizzata.
La base è il tipo "Qudit" già definito da ITensors, che rinomino in "Osc"; forse
è possibile usare direttamente le funzioni `op` sui Qudit definite da ITensors,
ma non riesco a trovare il modo, perciò sono costretto a definirle da capo.
La versione vettorizzata è un codice a parte, ma sempre ispirato alle stesse
funzioni.
La dimensione dell'oscillatore viene determinata dall'utente, al momento della
creazione del sito.
=#

# Matrici base 
# ------------
a⁻(dim::Int) = diagm(1 => [sqrt(j) for j = 1:dim-1])
a⁺(dim::Int) = diagm(-1 => [sqrt(j) for j = 1:dim-1])
num(dim::Int) = a⁺(dim) * a⁻(dim)
id(dim::Int) = Matrix{Int}(I, dim, dim)

function oscdimensions(length, basedim, decay)
  # Restituisce una sequenza decrescente da `basedim` a 2, da assegnare come
  # dimensione dello spazio di Hilbert della catena di oscillatori, in modo
  # che quelli più in profondità della catena non siano inutilmente grandi.
  f(j) = 2 + basedim * ℯ^(-decay * j)
  return [basedim; basedim; (Int ∘ floor ∘ f).(3:length)]
end

# Spazio degli oscillatori (normale)
# ==================================
alias(::SiteType"Osc") = SiteType"Qudit"()
ITensors.space(st::SiteType"Osc"; kwargs...) = space(alias(st); kwargs...)

# Stati
# -----
#=
Per creare uno stato della base canonica, si passa il numero di occupazione
sotto forma di stringa alla funzione `state`.
Ad esempio, se s è un sito ITensor di tipo "Osc",
  state(s, "n")
crea uno stato con n quanti nel sito s, mentre se `sites` è un array di m siti
  MPS(sites, ["n₁", "n₂", …, "nₘ"])
crea un MPS con il j-esimo sito occupato da nⱼ quanti.
=#
ITensors.state(sn::StateName, st::SiteType"Osc", s::Index) = state(sn, alias(st), s)

# Operatori
# ---------
#=
Quando si invoca la funzione
  op(name::AbstractString, s::Index...; kwargs...)
ITensor prova a chiamare le seguenti funzioni (in ordine, tra le altre):
  op(::OpName, ::SiteType, ::Index; kwargs...)
  op(::OpName, ::SiteType; kwargs...)
Definendo la seguente funzione, intercetto la prima chiamata, e posso far
calcolare la dimensione alla funzione, a partire dall'Index dato.
La funzione aggiungerà la dimensione ai kwargs, e a sua volta chiamerà una
delle `op` definite qui di seguito, trasformando il risultato in un oggetto
ITensor nel modo appropriato (vedere src/physics/site_types/qudit.jl alla
riga 73 come esempio).
=#
function ITensors.op(on::OpName, st::SiteType"Osc", s::Index; kwargs...)
  return itensor(op(on, st; dim=ITensors.dim(s), kwargs...), s', dag(s))
end

ITensors.op(::OpName"a+", ::SiteType"Osc"; dim=2) = a⁺(dim)
ITensors.op(::OpName"a-", ::SiteType"Osc"; dim=2) = a⁻(dim)
ITensors.op(::OpName"plus", st::SiteType"Osc"; kwargs...) = ITensors.op(OpName("a+"), st; kwargs...)
ITensors.op(::OpName"minus", st::SiteType"Osc"; kwargs...) = ITensors.op(OpName("a-"), st; kwargs...)

ITensors.op(::OpName"Id", ::SiteType"Osc"; dim=2) = id(dim)
ITensors.op(::OpName"N", ::SiteType"Osc"; dim=2) = num(dim)
ITensors.op(::OpName"X", ::SiteType"Osc"; dim=2) = a⁻(dim) + a⁺(dim)
ITensors.op(::OpName"Y", ::SiteType"Osc"; dim=2) = im*(a⁻(dim) - a⁺(dim))

# Spazio degli oscillatori vettorizzato
# =====================================
function ITensors.space(::SiteType"vecOsc"; dim=2)
  return dim^2
end

# Stati
# -----
#=
Tutti gli stati del tipo "vecOsc" variano in base alla dimensione dello spazio,
che è la radice quadrata della variabile `dim` degli Index di questo tipo.

Quando viene chiamata la funzione
  state(s::Index, name::AbstractString; kwargs...)
ITensors chiama nuovamente la funzione `state` provando diverse combinazioni
di argomenti, tra cui (in ordine)
  state(::StateName"Name", ::SiteType"Tag", s::Index; kwargs...)
  state(::StateName"Name", ::SiteType"Tag"; kwargs...)
e incapsulando il risultato in un oggetto ITensor.
La seconda, che deve restituire un vettore, è il tipo di funzione che
sovrascriverò con le definizioni date di seguito.

A questa funzione l'Index non viene passato come argomento, quindi devo passare
la dimensione in qualche modo tramite i kwargs.
La seguente definizione di `state` per il tipo "vecOsc" intercetta la prima
delle due combinazioni di cui sopra, calcola dall'Index fornito come argomento
la dimensione giusta, la aggiunge ai kwargs e chiama poi la seconda combinazione.
=#
function ITensors.state(sn::StateName, st::SiteType"vecOsc", s::Index; kwargs...)
  return state(sn, st; dim=isqrt(ITensors.dim(s)), kwargs...)
  # Uso `isqrt` che prende un Int e restituisce un Int; il fatto che tronchi la
  # parte decimale non dovrebbe mai essere un problema, dato che dim(s) per
  # costruzione è il quadrato di un intero.
end

#=
Prima di arrivare a chiamare `state` come spiegato qui sopra, però, ITensors
prova a chiamare anche 
  state(::StateName"Name", st::SiteType"Tag", s::Index; kwargs...)
per ogni `st` nei tag dell'Index `s`; se il risultato è `nothing` prova con il
tag successivo, fino ad esaurirli e poi continuare con altri tipi di argomenti
per `state`, se invece il risultato non è `nothing` lo restituisce, incapsulato
in un oggetto ITensor.
Qui c'è un problema: i siti creati con `siteinds` non hanno solo il tipo
"vecOsc" ma anche "Site" e "n=N", e non esiste alcuna funzione del tipo
  state(::StateName{ThermEq}, ::SiteType{Site}, ::Index{Int64}; kwargs...)
o con "n=N", di conseguenza Julia dà un errore di tipo MethodError.
Devo quindi definire questa funzione, e farle restituire `nothing` in modo
che venga saltata.
=#
ITensors.state(sn::StateName, st::SiteType, s::Index; kwargs...) = nothing

# Gli stati della base canonica (êₙ:êₙ) ≡ êₙ ⊗ êₙ
function ITensors.state(::StateName{N}, ::SiteType"vecOsc"; dim=2) where {N}
  n = parse(Int, String(N))
  v = zeros(dim)
  v[n + 1] = 1.0
  return v ⊗ v
end

# Lo stato di equilibrio termico Z⁻¹vec(exp(-βH)) = Z⁻¹vec(exp(-ω/T N))
function ITensors.state(::StateName"ThermEq", st::SiteType"vecOsc"; dim=2, ω, T)
  if T == 0
    v = state(StateName("0"), st; dim=dim)
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
    v = vcat(mat[:])
  end
  return v
end

# Stati del tipo êⱼ ⊗ êₖ (servono ad esempio per poter calcolare, tramite una
# proiezione su di essi, la componente j,k di una matrice definita su un sito
# di tipo "vecOsc").
function ITensors.state(::StateName"mat_comp", ::SiteType"vecOsc"; dim=2, j::Int, k::Int)
  êⱼ = zeros(dim)
  êₖ = zeros(dim)
  êⱼ[j + 1] = 1.0
  êₖ[k + 1] = 1.0
  return êⱼ ⊗ êₖ
end

# Stati che sono operatori vettorizzati (per costruire le osservabili)
# --------------------------------------------------------------------
function ITensors.state(::StateName"veca+", ::SiteType"vecOsc"; dim=2)
  return vec(a⁺(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"veca-", ::SiteType"vecOsc"; dim=2)
  return vec(a⁻(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"vecN", ::SiteType"vecOsc"; dim=2)
  return vec(num(dim), canonicalbasis(dim))
end
function ITensors.state(::StateName"vecId", ::SiteType"vecOsc"; dim=2)
  return vec(id(dim), canonicalbasis(dim))
end

function ITensors.state(::StateName"vecplus", st::SiteType"vecOsc"; kwargs...)
  return ITensors.state(StateName("veca+"), st; kwargs...)
end
function ITensors.state(::StateName"vecminus", st::SiteType"vecOsc"; kwargs...)
  return ITensors.state(StateName("veca-"), st; kwargs...)
end

# Operatori generici per oscillatori vettorizzati
# -----------------------------------------------
function ITensors.op(on::OpName, st::SiteType"vecOsc", s::Index; kwargs...)
  return itensor(op(on, st; dim=isqrt(ITensors.dim(s)), kwargs...), s', dag(s))
end
# Operatori semplici sullo spazio degli oscillatori
ITensors.op(::OpName"Id:Id", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ id(dim)
ITensors.op(::OpName"Id", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ id(dim)
# - interazione con la catena
ITensors.op(::OpName"Id:asum", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ (a⁺(dim)+a⁻(dim))
ITensors.op(::OpName"asum:Id", ::SiteType"vecOsc"; dim=2) = (a⁺(dim)+a⁻(dim)) ⊗ id(dim)
# - Hamiltoniano del sistema libero
ITensors.op(::OpName"N:Id", ::SiteType"vecOsc"; dim=2) = num(dim) ⊗ id(dim)
ITensors.op(::OpName"Id:N", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ num(dim)
# - termini di dissipazione
ITensors.op(::OpName"a-a+:Id", ::SiteType"vecOsc"; dim=2) = (a⁻(dim)*a⁺(dim)) ⊗ id(dim)
ITensors.op(::OpName"Id:a-a+", ::SiteType"vecOsc"; dim=2) = id(dim) ⊗ (a⁻(dim)*a⁺(dim))
ITensors.op(::OpName"a+T:a-", ::SiteType"vecOsc"; dim=2) = transpose(a⁺(dim)) ⊗ a⁻(dim)
ITensors.op(::OpName"a-T:a+", ::SiteType"vecOsc"; dim=2) = transpose(a⁻(dim)) ⊗ a⁺(dim)

# Spazio degli oscillatori vettorizzato hermitiano
# ================================================
function ITensors.space(::SiteType"HvOsc"; dim=2)
  return dim^2
end

# Stati
# -----
function ITensors.state(sn::StateName, st::SiteType"HvOsc", s::Index; kwargs...)
  return state(sn, st; dim=isqrt(ITensors.dim(s)), kwargs...)
  # Uso `isqrt` che prende un Int e restituisce un Int; il fatto che tronchi la
  # parte decimale non dovrebbe mai essere un problema, dato che dim(s) per
  # costruzione è il quadrato di un intero.
end

# Gli stati della base canonica (êₙ:êₙ) ≡ êₙ ⊗ êₙ
function ITensors.state(::StateName{N}, ::SiteType"HvOsc"; dim=2) where {N}
  n = parse(Int, String(N))
  v = zeros(dim)
  v[n + 1] = 1.0
  return vec(v ⊗ v', gellmannbasis(dim))
end

# Lo stato di equilibrio termico Z⁻¹vec(exp(-βH)) = Z⁻¹vec(exp(-ω/T N))
function ITensors.state(::StateName"ThermEq", st::SiteType"HvOsc"; dim=2, ω, T)
  if T == 0
    v = state(StateName("0"), st; dim=dim)
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
    v = vec(mat, gellmannbasis(dim))
  end
  return v
end

# Prodotto di X e dello stato eq. termico Z⁻¹vec(exp(-βH)) = Z⁻¹vec(exp(-ω/T N))
function ITensors.state(::StateName"X⋅Therm", st::SiteType"HvOsc"; dim=2, ω, T)
  if T == 0
    mat = zeros(Float64, dim, dim)
    mat[1, 1] = 1.0
  else
    mat = exp(-ω / T * num(dim))
    mat /= tr(mat)
  end
  return vec((a⁺(dim) + a⁻(dim)) * mat, gellmannbasis(dim))
end

# Stati del tipo êⱼ ⊗ êₖ (servono ad esempio per poter calcolare, tramite una
# proiezione su di essi, la componente j,k di una matrice definita su un sito
# di tipo "HvOsc").
function ITensors.state(::StateName"mat_comp", ::SiteType"HvOsc"; dim=2, j::Int, k::Int)
  êⱼ = zeros(dim)
  êₖ = zeros(dim)
  êⱼ[j + 1] = 1.0
  êₖ[k + 1] = 1.0
  return vec(êⱼ ⊗ êₖ', gellmannbasis(dim))
end

# Stati che sono operatori vettorizzati (per costruire le osservabili)
# --------------------------------------------------------------------
function ITensors.state(::StateName"veca+", ::SiteType"HvOsc"; dim=2)
  return vec(a⁺(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"veca-", ::SiteType"HvOsc"; dim=2)
  return vec(a⁻(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecN", ::SiteType"HvOsc"; dim=2)
  return vec(num(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecId", ::SiteType"HvOsc"; dim=2)
  return vec(id(dim), gellmannbasis(dim))
end

function ITensors.state(::StateName"vecX", ::SiteType"HvOsc"; dim=2)
  return vec(a⁻(dim) + a⁺(dim), gellmannbasis(dim))
end
function ITensors.state(::StateName"vecY", ::SiteType"HvOsc"; dim=2)
  return vec(im*(a⁻(dim) - a⁺(dim)), gellmannbasis(dim))
end

function ITensors.state(::StateName"vecplus", st::SiteType"HvOsc"; kwargs...)
  return ITensors.state(StateName("veca+"), st; kwargs...)
end
function ITensors.state(::StateName"vecminus", st::SiteType"HvOsc"; kwargs...)
  return ITensors.state(StateName("veca-"), st; kwargs...)
end

# Operatori generici per oscillatori vettorizzati
# -----------------------------------------------
function ITensors.op(on::OpName, st::SiteType"HvOsc", s::Index; kwargs...)
  return itensor(op(on, st; dim=isqrt(ITensors.dim(s)), kwargs...), s', dag(s))
end

function ITensors.op(::OpName"Id", ::SiteType"HvOsc"; dim=2)
  return vec(identity, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅a+", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*a⁺(dim), gellmannbasis(dim))
end
function ITensors.op(::OpName"a+⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> a⁺(dim)*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅a-", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*a⁻(dim), gellmannbasis(dim))
end
function ITensors.op(::OpName"a-⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> a⁻(dim)*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"⋅asum", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*(a⁺(dim)+a⁻(dim)), gellmannbasis(dim))
end
function ITensors.op(::OpName"asum⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> (a⁺(dim)+a⁻(dim))*x, gellmannbasis(dim))
end

function ITensors.op(::OpName"N⋅", ::SiteType"HvOsc"; dim=2)
  return vec(x -> num(dim)*x, gellmannbasis(dim))
end
function ITensors.op(::OpName"⋅N", ::SiteType"HvOsc"; dim=2)
  return vec(x -> x*num(dim), gellmannbasis(dim))
end

# Termini nell'equazione di Lindblad per oscillatori vettorizzati
# ---------------------------------------------------------------
# Termini di smorzamento
function ITensors.op(::OpName"Damping", ::SiteType"vecOsc", s::Index; ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  d = (n + 1) * (op("a+T:a-", s) - 0.5 * (op("N:Id", s) + op("Id:N", s))) +
      n * (op("a-T:a+", s) - 0.5 * (op("a-a+:Id", s) + op("Id:a-a+", s)))
  return d
end
function ITensors.op(::OpName"Damping", ::SiteType"HvOsc"; dim=2, ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> (n + 1) * (A*x*A⁺ - 0.5*(A⁺*A*x + x*A⁺*A)) +
               n * (A⁺*x*A - 0.5*(A*A⁺*x + x*A*A⁺)),
          gellmannbasis(dim))
  return d
end

function ITensors.op(::OpName"Lindb+", ::SiteType"HvOsc"; dim=2)
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> A*x*A⁺ - 0.5*(A⁺*A*x + x*A⁺*A), gellmannbasis(dim))
  return d
end
function ITensors.op(::OpName"Lindb-", ::SiteType"HvOsc"; dim=2)
  A = a⁻(dim)
  A⁺ = a⁺(dim)
  d = vec(x -> A⁺*x*A - 0.5*(A*A⁺*x + x*A*A⁺),
          gellmannbasis(dim))
  return d
end

function mixedlindbladplus(s1::Index{Int64}, s2::Index{Int64})
  return (op("a-⋅", s1) * op("⋅a+", s2) +
          op("a-⋅", s2) * op("⋅a+", s1) -
          0.5*(op("a+⋅", s1) * op("a-⋅", s2) +
               op("a+⋅", s2) * op("a-⋅", s1)) -
          0.5*(op("⋅a+", s1) * op("⋅a-", s2) +
               op("⋅a+", s2) * op("⋅a-", s1)))
end
function mixedlindbladminus(s1::Index{Int64}, s2::Index{Int64})
  return (op("a+⋅", s1) * op("⋅a-", s2) +
          op("a+⋅", s2) * op("⋅a-", s1) -
          0.5*(op("a-⋅", s1) * op("a+⋅", s2) +
               op("a-⋅", s2) * op("a+⋅", s1)) -
          0.5*(op("⋅a-", s1) * op("⋅a+", s2) +
               op("⋅a-", s2) * op("⋅a+", s1)))
end
# Proiezione sugli autostati dell'operatore numero
# ------------------------------------------------
# Il sito è uno solo quindi basta usare i vettori della base canonica
function osc_levels_proj(site::Index{Int64}, level::Int)
  st = state(site, "$level")
  return MPS([st])
end

# Scelta dello stato iniziale dell'oscillatore
# --------------------------------------------
# Con un'apposita stringa nei parametri è possibile scegliere lo stato da cui
# far partire l'oscillatore. La seguente funzione traduce la stringa
# nell'MPS desiderato, in modo case-insensitive. Le opzioni sono:
# · "thermal": stato di equilibrio termico
# · "fockN": autostato dell'operatore numero con N quanti di eccitazione
# · "empty": alias per "fock0"
function parse_init_state_osc(site::Index{Int64}, statename::String; kwargs...)
  statename = lowercase(statename)
  if statename == "thermal"
    s = state(site, "ThermEq"; kwargs...)
  elseif occursin(r"^fock", statename)
    j = parse(Int, replace(statename, "fock" => ""))
    s = state(site, "$j")
  elseif statename == "empty"
    s = state(site, "0")
  else
    throw(DomainError(statename,
                      "Stato non riconosciuto; scegliere tra «empty», «fockN» "*
                      "oppure «thermal»."))
  end
  return MPS([s])
end
