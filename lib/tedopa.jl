using PolyChaos
using QuadGK

function thermalisedJ(J::Function,
                      ω::Real,
                      T::Real)
  # Questa funzione restituisce la densità spettrale termalizzata, a partire da
  # una J(ω) già antisimmetrica attorno a ω=0 (se non lo è, è necessario farlo
  # prima di chiamare questa funzione).
  #
  # Calcolo il "fattore termico" con cui moltiplicare J; se T=0, per evitare
  # divisioni per zero, scrivo a parte il fattore: lim T->0 coth(x/T)=Θ(x).
  # Si potrebbe anche scrivere f=1 e basta, assumendo che chi chiama la funzione
  # abbia usato un supporto che escluda già (-∞,0), ma così almeno sto sicuro.
  if T == 0
    f = 0.5(1 + sign(ω))
  else
    f = 0.5(1 + coth(0.5 * ω / T))
  end
  return f * J(ω)
end

function chainmapcoefficients(J::Function,
                              support,
                              L::Int;
                              kwargs...)
  #=
  Calcola i coefficienti dell'Hamiltoniana ottenuta dalla "chain map", a
  partire da una data funzione spettrale.

  Argomenti
  ---------
  · `J::Function`: una funzione dall'intervallo specificato in `support` a 
    valori non negativi.

  · `support`: il supporto della funzione; al minimo è una coppia di numeri
    reali, ma può anche essere una lista di numeri reali (in ordine
    crescente), in tal caso il supporto è interpretato come l'unione degli
    intervalli con estremi i numeri dati. Ad esempio (0,2,3,4) individua il
    supporto (0,2)∪(2,3)∪(3,4). Questa suddivisione del supporto è del
    tutto ignorata per calcolare i coefficienti di ricorsione, ma può
    tornare utile per calcolare l'integrale di J, dato che a volte
    l'algoritmo di integrazione ha bisogno di questi punti intermedi per
    funzionare.

  · `L::Int`: la lunghezza della serie di oscillatori che si desidera
    costruire, nel senso del numero di oscillatori presenti.

  · `Nquad::Int`: un parametro da passare ad OrthoPoly, che specifica il numero
    di nodi usati nel metodo delle quadrature per calcolare gli integrali. Per
    modificare la precisione dei risultati, modificare questo parametro.

  I valori αₙ e βₙ sono i coefficienti nella formula di ricorsione
  x πₙ(x) = πₙ₊₁(x) + αₙ πₙ(x) + βₙ πₙ₋₁(x)
  dove {πₙ}ₙ, con n ∈ ℕ è la sequenza di polinomi ortogonali monici
  associata alla funzione J; π₋₁ è il polinomio nullo.
  
  Per costruire una serie di L oscillatori, servono i coefficienti αᵢ e βᵢ da
  i=0 a i=L-1: siccome però gli array sono indicizzati partendo da 1 in Julia,
  mi serviranno α[1:L] e β[1:L].
  A partire da questi coefficienti, indicizzando i siti degli oscillatori da 1
  a L, le frequenze Ωᵢ di ciascun oscillatore e la costante di interazione κᵢ
  tra gli oscillatori (i,i+1) sono date da
  Ωᵢ = αᵢ            per i∈{1,…,L},
  κᵢ = sqrt(βᵢ₊₁)    per i∈{1,…,L-1}.
  Il coefficiente β₀ rimane inutilizzato (onestamente, siccome a livello
  teorico è arbitrario, il suo valore dipende dall'implementazione di
  PolyChaos, quindi non so neanche quanto vale).
  Rimane il coefficiente η di interazione tra oscillatore e spin, che calcolo
  come l'integrale di J sul supporto dato.
  =#
  measure = PolyChaos.Measure("measure", J, (support[begin], support[end]), false, Dict())
  poly = PolyChaos.OrthoPoly("poly", L, measure; kwargs...)
  α = coeffs(poly)[:,1]
  β = coeffs(poly)[:,2]
  Ω = α
  κ = sqrt.(β[2:end])
  η = sqrt(quadgk(J, support...)[begin])
  return Ω, κ, η
end
