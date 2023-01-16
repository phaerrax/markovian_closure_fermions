using LinearAlgebra
using ITensors
using Combinatorics

include("utils.jl")

const ⊗ = kron

# Matrici di Pauli e affini
# -------------------------
σˣ = [0 1; 1 0]
σʸ = [0 -im; im 0]
σᶻ = [1 0; 0 -1]
σ⁺ = [0 1; 0 0]
σ⁻ = [0 0; 1 0]
I₂ = [1 0; 0 1]
ê₊ = [1; 0]
ê₋ = [0; 1]

# Il vettore nullo:
ITensors.state(::StateName"0", ::SiteType"S=1/2") = [0; 0]
# Definisco l'operatore numero per un singolo spin, che altro non è che
# la proiezione sullo stato |↑⟩.
ITensors.op(::OpName"N", st::SiteType"S=1/2") = op(OpName"ProjUp"(), st)
# La matrice identità
ITensors.op(::OpName"Id", ::SiteType"S=1/2") = I₂
# La matrice nulla
ITensors.op(::OpName"0", ::SiteType"S=1/2") = zeros(2, 2)
# Matrici di Pauli
# Operatori di scala
ITensors.op(::OpName"σ+", st::SiteType"S=1/2") = op(OpName("S+"), st)
ITensors.op(::OpName"σ-", st::SiteType"S=1/2") = op(OpName("S-"), st)
ITensors.op(::OpName"plus", st::SiteType"S=1/2") = op(OpName("S+"), st)
ITensors.op(::OpName"minus", st::SiteType"S=1/2") = op(OpName("S-"), st)

# Spazio degli spin vettorizzato
# ==============================
ITensors.space(::SiteType"vecS=1/2") = 4

# Stati (veri e propri)
# ---------------------
ITensors.state(::StateName"Up", ::SiteType"vecS=1/2") = ê₊ ⊗ ê₊
ITensors.state(::StateName"Dn", ::SiteType"vecS=1/2") = ê₋ ⊗ ê₋

# Stati che sono operatori vettorizzati (per costruire le osservabili)
# --------------------------------------------------------------------
function ITensors.state(::StateName"vecσx", ::SiteType"vecS=1/2")
  return vec(σˣ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecσy", ::SiteType"vecS=1/2")
  return vec(σʸ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecσz", ::SiteType"vecS=1/2")
  return vec(σᶻ, canonicalbasis(2))
end
function ITensors.state(::StateName"vecId", ::SiteType"vecS=1/2")
  return vec(I₂, canonicalbasis(2))
end
function ITensors.state(::StateName"vecN", ::SiteType"vecS=1/2")
  return vec([1 0; 0 0], canonicalbasis(2))
end
function ITensors.state(::StateName"vec0", ::SiteType"vecS=1/2")
  return vec([0 0; 0 0], canonicalbasis(2))
end

function ITensors.state(::StateName"vecX", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσx"), st)
end
function ITensors.state(::StateName"vecY", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσy"), st)
end
function ITensors.state(::StateName"vecZ", st::SiteType"vecS=1/2")
  return ITensors.state(StateName("vecσz"), st)
end

function ITensors.state(::StateName"vecplus", ::SiteType"vecS=1/2")
  return vec(σ⁺, canonicalbasis(2))
end
function ITensors.state(::StateName"vecminus", ::SiteType"vecS=1/2")
  return vec(σ⁻, canonicalbasis(2))
end

# Operatori
# ---------
# - identità
ITensors.op(::OpName"Id:Id", ::SiteType"vecS=1/2") = I₂ ⊗ I₂
ITensors.op(::OpName"Id", ::SiteType"vecS=1/2") = I₂ ⊗ I₂
# - termini per l'Hamiltoniano locale
ITensors.op(::OpName"σz:Id", ::SiteType"vecS=1/2") = σᶻ ⊗ I₂
ITensors.op(::OpName"Id:σz", ::SiteType"vecS=1/2") = I₂ ⊗ σᶻ
# - termini per l'Hamiltoniano bilocale
ITensors.op(::OpName"Id:σ+", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁺
ITensors.op(::OpName"σ+:Id", ::SiteType"vecS=1/2") = σ⁺ ⊗ I₂
ITensors.op(::OpName"Id:σ-", ::SiteType"vecS=1/2") = I₂ ⊗ σ⁻
ITensors.op(::OpName"σ-:Id", ::SiteType"vecS=1/2") = σ⁻ ⊗ I₂
# - termini di smorzamento
ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = σˣ ⊗ σˣ
ITensors.op(::OpName"σx:Id", ::SiteType"vecS=1/2") = σˣ ⊗ I₂
ITensors.op(::OpName"Id:σx", ::SiteType"vecS=1/2") = I₂ ⊗ σˣ
ITensors.op(::OpName"Damping", ::SiteType"vecS=1/2") = (σˣ ⊗ σˣ) - (I₂ ⊗ I₂)
function ITensors.op(::OpName"Damping2", ::SiteType"vecS=1/2"; ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  d = vec(x -> (n + 1) * (σ⁻*x*σ⁺ - 0.5*(σ⁺*σ⁻*x + x*σ⁺*σ⁻)) +
               n * (σ⁺*x*σ⁻ - 0.5*(σ⁻*σ⁺*x + x*σ⁻*σ⁺)),
          canonicalbasis(2))
  return d
end

# Spazio degli spin vettorizzato (su base di Gell-Mann)
# =====================================================
ITensors.space(::SiteType"HvS=1/2") = 4
#=
Qui sviluppo matrici ed operatori sulla base {Λᵢ}ᵢ₌₁⁴ delle matrici di
Gell-Mann generalizzate (più il multiplo dell'identità).
Il vettore v delle coordinate di una matrice A ∈ Mat(ℂ²) ha come elementi
vᵢ = tr(Λᵢ * A)
mentre un operatore lineare L : Mat(ℂ²) → Mat(ℂ²) ha
ℓᵢⱼ = tr(Λᵢ * L(Λⱼ))
come matrice rappresentativa.
=#

# Stati (veri e propri)
# ---------------------
# "Up" ≡ ê₊ ⊗ ê₊'
# "Dn" ≡ ê₋ ⊗ ê₋'
function ITensors.state(::StateName"Up", ::SiteType"HvS=1/2")
  return vec(ê₊ ⊗ ê₊', gellmannbasis(2))
end
function ITensors.state(::StateName"Dn", ::SiteType"HvS=1/2")
  return vec(ê₋ ⊗ ê₋', gellmannbasis(2))
end

# Stati che sono operatori vettorizzati (per costruire le osservabili)
# --------------------------------------------------------------------
function ITensors.state(::StateName"vecσx", ::SiteType"HvS=1/2")
  return vec(σˣ, gellmannbasis(2))
end
function ITensors.state(::StateName"vecσy", ::SiteType"HvS=1/2")
  return vec(σʸ, gellmannbasis(2))
end
function ITensors.state(::StateName"vecσz", ::SiteType"HvS=1/2")
  return vec(σᶻ, gellmannbasis(2))
end

function ITensors.state(::StateName"vecId", ::SiteType"HvS=1/2")
  return vec(I₂, gellmannbasis(2))
end
function ITensors.state(::StateName"vecN", ::SiteType"HvS=1/2")
  return vec([1 0; 0 0], gellmannbasis(2))
end
function ITensors.state(::StateName"vec0", ::SiteType"HvS=1/2")
  return vec(zeros(2, 2), gellmannbasis(2))
end

function ITensors.state(::StateName"vecX", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσx"), st)
end
function ITensors.state(::StateName"vecY", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσy"), st)
end
function ITensors.state(::StateName"vecZ", st::SiteType"HvS=1/2")
  return ITensors.state(StateName("vecσz"), st)
end

function ITensors.state(::StateName"vecplus", ::SiteType"HvS=1/2")
  return vec(σ⁺, gellmannbasis(2))
end
function ITensors.state(::StateName"vecminus", ::SiteType"HvS=1/2")
  return vec(σ⁻, gellmannbasis(2))
end

# Operatori
# ---------
# Gli operatori, anche se agiscono su due siti contemporaneamente, sono
# sempre fattorizzati (o somme di operatori fattorizzati): se l'operatore
# L : Mat(ℂ²)⊗Mat(ℂ²) → Mat(ℂ²)⊗Mat(ℂ²)
# si scrive come L₁⊗L₂ con ciascun Lᵢ : Mat(ℂ²) → Mat(ℂ²) allora
# ⟨êᵢ₁ ⊗ êᵢ₂, L(êⱼ₁ ⊗ êⱼ₂)⟩ = ⟨êᵢ₁, L₁(êⱼ₁)⟩ ⟨êᵢ₂, L₂(êⱼ₂)⟩.
function ITensors.op(::OpName"Id", ::SiteType"HvS=1/2")
  return vec(identity, gellmannbasis(2))
end

function ITensors.op(s::OpName"σ+⋅", ::SiteType"HvS=1/2")
  return vec(x -> σ⁺*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σ+", ::SiteType"HvS=1/2")
  return vec(x -> x*σ⁺, gellmannbasis(2))
end

function ITensors.op(s::OpName"σ-⋅", ::SiteType"HvS=1/2")
  return vec(x -> σ⁻*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σ-", ::SiteType"HvS=1/2")
  return vec(x -> x*σ⁻, gellmannbasis(2))
end

function ITensors.op(s::OpName"σx⋅", ::SiteType"HvS=1/2")
  return vec(x -> σˣ*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σx", ::SiteType"HvS=1/2")
  return vec(x -> x*σˣ, gellmannbasis(2))
end

function ITensors.op(s::OpName"σz⋅", ::SiteType"HvS=1/2")
  return vec(x -> σᶻ*x, gellmannbasis(2))
end
function ITensors.op(::OpName"⋅σz", ::SiteType"HvS=1/2")
  return vec(x -> x*σᶻ, gellmannbasis(2))
end

# Operatori della corrente di spin
# ================================
# Jₖ,ₖ₊₁ = -λ/2 (σˣ₍ₖ₎σʸ₍ₖ₊₁₎ - σʸ₍ₖ₎σˣ₍ₖ₊₁₎)
function J⁺tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # Questa funzione restituisce i nomi degli operatori da assegnare al
  # sito i-esimo per la parte σˣ⊗σʸ di Jₖ,ₖ₊₁ (k ≡ left_site)
  if i == left_site
    str = "σx"
  elseif i == left_site + 1
    str = "σy"
  else
    str = "Id"
  end
  return str
end
function J⁺tag(::SiteType"vecS=1/2", left_site::Int, i::Int)
  if i == left_site
    str = "vecσx"
  elseif i == left_site + 1
    str = "vecσy"
  else
    str = "vecId"
  end
  return str
end
function J⁺tag(::SiteType"HvS=1/2", left_site::Int, i::Int)
  return J⁺tag(SiteType("vecS=1/2"), left_site, i)
end

function J⁻tag(::SiteType"S=1/2", left_site::Int, i::Int)
  # Come `J⁺tag`, ma per σʸ⊗σˣ
  if i == left_site
    str = "σy"
  elseif i == left_site + 1
    str = "σx"
  else
    str = "Id"
  end
  return str
end
function J⁻tag(::SiteType"vecS=1/2", left_site::Int, i::Int)
  if i == left_site
    str = "vecσy"
  elseif i == left_site + 1
    str = "vecσx"
  else
    str = "vecId"
  end
  return str
end
function J⁻tag(::SiteType"HvS=1/2", left_site::Int, i::Int)
  return J⁻tag(SiteType("vecS=1/2"), left_site, i)
end

function spin_current_op_list(sites::Vector{Index{Int64}})
  N = length(sites)
  # Controllo che i siti forniti siano degli spin ½.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    st = SiteType("S=1/2")
    MPtype = MPO
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites))
    st = SiteType("vecS=1/2")
    MPtype = MPS
  elseif all(x -> SiteType("HvS=1/2") in x, sitetypes.(sites))
    st = SiteType("HvS=1/2")
    MPtype = MPS
  else
    throw(ArgumentError("spin_current_op_list è disponibile per siti di tipo "*
                        "\"S=1/2\", \"vecS=1/2\" oppure \"HvS=1/2\"."))
  end
  #
  J⁺ = [MPtype(sites, [J⁺tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  J⁻ = [MPtype(sites, [J⁻tag(st, k, i) for i = 1:N]) for k = 1:N-1]
  return -0.5 .* (J⁺ .- J⁻)
end


# Base di autostati per la catena
# -------------------------------
#=
Come misurare il "contributo" di ogni autospazio dell'operatore numero
N su uno stato rappresentato dalla matrice densità ρ?
Se Pₙ è la proiezione sull'autospazio dell'autovalore n di N, il
valore cercato è cₙ = tr(ρ Pₙ), oppure cₙ = (ψ,Pₙψ⟩ a seconda di com'è
descritto lo stato del sistema.
Il proiettore Pₙ lo posso costruire sommando i proiettori su ogni singolo
autostato nell'n-esimo autospazio: è un metodo grezzo ma dovrebbe
funzionare. Calcolando prima dell'avvio dell'evoluzione temporale tutti
i proiettori si dovrebbe risparmiare tempo, dovento effettuare poi solo
n_sites operazioni ad ogni istante di tempo.                         

La seguente funzione restituisce i nomi per costruire i MPS degli
stati della base dell'intero spazio della catena (suddivisi per
numero di occupazione complessivo).
=#
function chain_basis_states(n_sites::Int, level::Int)
  return unique(permutations([repeat(["Up"], level);
                              repeat(["Dn"], n_sites - level)]))
end

# La seguente funzione crea un proiettore su ciascun sottospazio con
# livello di occupazione definito.
# Metodo "crudo": prendo tutti i vettori della base ortonormale di
# ciascun autospazio, ne creo i proiettori ortogonali e li sommo tutti.
# Forse poco efficiente, ma funziona.
function level_subspace_proj(sites::Vector{Index{Int64}}, level::Int)
  N = length(sites)
  # Controllo che i siti forniti siano degli spin ½.
  if all(x -> SiteType("S=1/2") in x, sitetypes.(sites))
    projs = [projector(MPS(sites, names); normalize=false)
             for names in chain_basis_states(N, level)]
  elseif all(x -> SiteType("vecS=1/2") in x, sitetypes.(sites)) ||
         all(x -> SiteType("HvS=1/2") in x, sitetypes.(sites))
    projs = [MPS(sites, names)
             for names in chain_basis_states(N, level)]
  else
    throw(ArgumentError("spin_current_op_list è disponibile per siti di tipo "*
                        "\"S=1/2\", \"vecS=1/2\" oppure \"HvS=1/2\"."))
  end
  #return sum(projs) non funziona…
  P = projs[1]
  for p in projs[2:end]
    P = +(P, p; cutoff=1e-10)
  end
  return P
end


# Autostati del primo livello
# ---------------------------
# Potrebbe essere utile anche avere a disposizione la forma degli autostati della
# catena di spin (isolata).
# Gli stati sₖ che hanno il sito k nello stato eccitato e gli altri nello stato
# fondamentale infatti non sono autostati; nella base {sₖ}ₖ (k=1:N) posso
# scrivere l'Hamiltoniano della catena isolata, ristretto all'autospazio di
# singola eccitazione, come
# ε λ 0 0 … 0
# λ ε λ 0 … 0
# 0 λ ε λ … 0
# 0 0 λ ε … 0
# … … … … … …
# 0 0 0 0 … ε
# che ha come autostati vⱼ= ∑ₖ sin(kjπ /(N+1)) sₖ, con j=1:N.
# Attenzione poi a normalizzarli: ‖vⱼ‖² = (N+1)/2.
function single_ex_state(sites::Vector{Index{Int64}}, k::Int)
  N = length(sites)
  if k ∈ 1:N
    states = [i == k ? "Up" : "Dn" for i ∈ 1:N]
  else
    throw(DomainError(k,
                      "Si è tentato di costruire uno stato con eccitazione "*
                      "localizzata nel sito $k, che non appartiene alla "*
                      "catena: inserire un valore tra 1 e $N."))
  end
  return MPS(sites, states)
end

function chain_L1_state(sites::Vector{Index{Int64}}, j::Int)
  # Occhio ai coefficienti: questo, come sopra, non è il vettore vⱼ ma è
  # vec(vₖ⊗vₖᵀ) = vₖ⊗vₖ: di conseguenza i coefficienti della combinazione
  # lineare qui sopra devono essere usati al quadrato.
  # Il prodotto interno tra matrici vettorizzate è tale che
  # ⟨vec(a⊗aᵀ),vec(b⊗bᵀ)⟩ = tr((a⊗aᵀ)ᵀ b⊗bᵀ) = (aᵀb)².
  # Non mi aspetto che la norma di questo MPS sia 1, pur avendo
  # usato i coefficienti normalizzati. Quello che deve fare 1 è invece la
  # somma dei prodotti interni di vec(sₖ⊗sₖ), per k=1:N, con questo vettore.
  # Ottengo poi che ⟨Pₙ,vⱼ⟩ = 1 solo se n=j, 0 altrimenti.
  N = length(sites)
  if j ∉ 1:N
    throw(DomainError(j,
                      "Si è tentato di costruire un autostato della catena "*
                      "con indice $j, che non è valido: bisogna fornire un "*
                      "indice tra 1 e $N."))
  end
  states = [2/(N+1) * sin(j*k*π / (N+1))^2 * single_ex_state(sites, k)
            for k ∈ 1:N]
  return sum(states)
end

# Scelta dello stato iniziale della catena
# ----------------------------------------
# Con un'apposita stringa nei parametri è possibile scegliere lo stato da cui
# far partire la catena di spin. La seguente funzione traduce la stringa
# nell'MPS desiderato, in modo case-insensitive. Le opzioni sono:
# · "empty": stato vuoto
# · "1locM": stato con una (sola) eccitazione localizzata nel sito M ∈ {1,…,N}
# · "1eigM": autostato del primo livello con M ∈ {0,…,N-1} nodi
function parse_init_state(sites::Vector{Index{Int64}}, state::String)
  state = lowercase(state)
  if state == "empty"
    v = MPS(sites, "Dn")
  elseif occursin(r"^1loc", state)
    j = parse(Int, replace(state, "1loc" => ""))
    v = single_ex_state(sites, j)
  elseif occursin(r"^1eig", state)
    j = parse(Int, replace(state, "1eig" => ""))
    # Il j-esimo autostato ha j-1 nodi 
    v = chain_L1_state(sites, j + 1)
  else
    throw(DomainError(state,
                      "Stato non riconosciuto; scegliere tra «empty», «1locN» "*
                      "oppure «1eigN»."))
  end
  return v
end

# La seguente funzione è come quella sopra, ma per uno spin solo.
function parse_spin_state(site::Index{Int64}, state::String)
  state = lowercase(state)
  if state == "empty" || state == "dn" || state == "down"
    v = ITensors.state(site, "Dn")
  elseif state == "up"
    v = ITensors.state(site, "Up")
  elseif state == "x+"
    v = 1/sqrt(2) * (ITensors.state(site, "Up") + ITensors.state(site, "Dn"))
  else
    throw(DomainError(state,
                      "Stato non riconosciuto; scegliere tra «empty», «up», "*
                      "«down» oppure «x+»."))
  end
  return MPS([v])
end
