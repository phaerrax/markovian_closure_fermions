using JSON
using Base.Filesystem
using ITensors
using LinearAlgebra

#= Questo file contiene un insieme variegato di funzioni che servono bene o
   male per tutti gli script. Le raduno tutte qui in modo da non dover copiare
   il loro codice ad ogni nuovo file.
=#

# Vettorizzazione di matrici ed operatori
# =======================================
# L'uso di tr(x'*y) come prodotto interno tra gli elementi x e y fa sì che
# queste formule di vettorizzazione abbiano senso solo quando la base scelta
# è ortonormale rispetto a tale prodotto interno: vanno bene ad esempio la
# base canonica o quella di Gell-Mann.
function vec(A::Matrix, basis::Vector)
  return [tr(b' * A) for b ∈ basis]
end
function vec(L::Function, basis::Vector)
  return [tr(bi' * L(bj)) for (bi, bj) ∈ Base.product(basis, basis)]
end

function partialtrace(sites::Vector{Index{Int64}}, v::MPS, j::Int)
  # Calcola la traccia parziale della matrice rappresentata (tramite
  # vettorizzazione) dal MPS `v` nel j° sito.
  # (Restituisce la matrice ancora nella forma vettorizzata.)
  # Contraggo tutti i siti in v[2:end] con uno stato vecId per ottenere
  # un MPS con un solo indice fisico: quello dello stato a sinistra,
  # che diventa così un vettore.
  # Definisco una procedura di contrazione personalizzata: parto da v[end]
  # e lo contraggo per prima cosa con un vecId; il risultato lo contraggo
  # con v[end-1], poi con un altro vecId ma sul sito end-1, e così via fino
  # ad arrivare a v[j].
  # Poi faccio lo stesso partendo da v[1] e contraendo verso destra.
  x = ITensor(1.)
  for i ∈ length(sites):-1:j+1
    x = v[i] * state(sites[i], "vecId") * x
  end
  y = ITensor(1.)
  for i ∈ 1:j-1
    y = v[i] * state(sites[i], "vecId") * y
  end
  z = y * v[j] * x
  # Ora `vector(z)` è il vettore di coordinate della traccia parziale.
  return vector(z)
end

# Matrici di Gell-Mann generalizzate
# ==================================
# Le matrici sono qui indicizzate come segue:
# · se j > k restituisce le matrici simmetriche con diagonale nulla,
# · se j < k restituisce le matrici antisimmetriche,
# · se j = k restituisce le matrici diagonali.
# Gli interi j e k determinano anche le componenti che non sono nulle.
# Il totale delle matrici di Gell-Mann generalizzate è dim²-1 come richiesto;
# aggiungo alla fine (per il caso j = k = dim) la matrice identità, e completo
# la base per Mat(ℂᵈⁱᵐ).
# Le matrici sono normalizzate in modo da essere ortonormali rispetto al
# prodotto interno (A,B) = tr(A†B)
function gellmannmatrix(j, k, dim)
  if j > dim || k > dim || j < 0 || k < 0
    throw(DomainError)
  end
  m = zeros(ComplexF64, dim, dim)
  if j > k
    m[j, k] = 1 / sqrt(2)
    m[k, j] = 1 / sqrt(2)
  elseif k > j
    m[j, k] = -im / sqrt(2)
    m[k, j] = im / sqrt(2)
  elseif j == k && j < dim
    for i ∈ 1:j
      m[i, i] = 1
    end
    m[j+1, j+1] = -j
    m .*= sqrt(1 / (j * (j+1)))
  else
    for i ∈ 1:dim
      m[i, i] = 1/sqrt(dim)
    end
  end
  return m
end

function gellmannbasis(dim)
  return [gellmannmatrix(j, k, dim)
          for (j, k) ∈ [Base.product(1:dim, 1:dim)...]]
  # Devo spacchettare il risultato di `product` in modo che il risultato
  # sia una lista di matrici (un Vector, per la precisione) e non una Matrix.
end

# Inserisco anche una definizione della base canonica di Mat(ℂᵈⁱᵐ): non mi serve
# tanto perché è una costruzione difficile, ma perché voglio impostare l'ordine
# in cui appaiono gli elementi della base. Nel modo in cui lo faccio sotto,
# l'ordine corrisponde a come si fa la vettorizzazione delle matrici, cioè
# canonicalbasis(dim)[j] = canonicalmatrix((j-1)%dim + 1, (j-1)÷dim + 1, dim)
# con j ∈ {1,…,dim²}. Di conseguenza, se A ∈ Mat(ℂᵈⁱᵐ), vale
# vec(A)ᵢ = tr(canonicalbasis(dim)[j]' * A).
function canonicalmatrix(i, j, dim)
  m = zeros(ComplexF64, dim, dim)
  m[i,j] = 1
  return m
end

function canonicalbasis(dim)
  return [canonicalmatrix(i, j, dim)
          for (i, j) ∈ [Base.product(1:dim, 1:dim)...]]
end

# Entropia (di Von Neumann)
# =========================
function vonneumannentropy(ψ::MPS, sites::Vector{Index{Int64}}, n::Int)
  # Calcola l'entropia di entanglement della bipartizione (1,…,n)|(n+1,…,N)
  # del sistema nello stato ψ usando la decomposizione di Schmidt.
  orthogonalize!(ψ, n)
  # Decomponi ψ[n] nei suoi valori singolari, trattando il Link tra il sito
  # n-1 e il sito n e l'indice fisico come "indici di riga"; il Link tra il
  # sito n e il sito n+1, che rimane, sarà l'"indice di colonna".
  _, S, _ = svd(ψ[n], (linkind(ψ, n-1), sites[n]))
  # Calcolo il quadrato dei valori singolari (aka i coefficienti di Schmidt
  # della bipartizione), e infine da essi l'entropia.
  sqdiagS = [S[j,j]^2 for j ∈ dim(S, 1)]
  return -sum(p -> p * log(p), sqdiagS; init=0.0)
end

# Chop da Mathematica
# ===================
# Imitazione della funzione "Chop" di Mathematica. Tronca l'argomento a zero
# se è al di sotto di una data soglia.
function chop(x::Real; tolerance=1e-10)
  return abs(x) > tolerance ? x : zero(x)
end

function chop(x::Complex; tolerance=1e-10)
  return Complex(chop(real(x)), chop(imag(x)))
end

# Manipolazione dei dati di oggetti di ITensors
# =============================================
function sitetypes(s::Index)
  # Questa funzione estrae i tag dell'oggetto Index fornito, e costruisce per
  # ciascuno un oggetto SiteType. Questi SiteType possono poi essere interpretati
  # da una funzione che riceve l'Index come argomento per capire come comportarsi.
  # [La funzione è già presente nella libreria ITensors, ma è solo interna, quindi
  # devo ricopiarla qui per poterla utilizzare.]
  ts = tags(s)
  return SiteType[SiteType(ts.data[n]) for n in 1:length(ts)]
end

# Lettura dei parametri della simulazione
# =======================================
function isjson(filename::String)
  return length(filename) > 5 && filename[end-4:end] == ".json"
  # Volutamente, un file che di nome fa solo ".json" viene ignorato.
end

function load_parameters(file_list)
  if isempty(file_list)
    throw(ErrorException("Non è stato fornito alcun file di parametri."))
  end
  first_arg = file_list[1]
  prev_dir = pwd()
  if isdir(first_arg)
    # Se il primo input è una cartella, legge tutti i file .json al suo
    # interno e li carica come file di parametri; alla fine i file di
    # output saranno salvati in tale cartella.
    # I restanti elementi di ARGS vengono ignorati (l'utente viene avvisato).
    cd(first_arg)
    files = filter(isjson, readdir())
    @info "$first_arg è una cartella. I restanti argomenti passati alla "*
          "linea di comando saranno ignorati."
  else
    # Se il primo input non è una cartella, tutti gli argomenti passati in
    # ARGS vengono trattati come file da leggere contenenti i parametri.
    # I file di output saranno salvati nella cartella pwd() al momento
    # del lancio del programma.
    files = file_list
  end
  # Carico i file di parametri nei dizionari.
  parameter_lists = []
  for f in files
    open(f) do input
      s = read(input, String)
      # Aggiungo anche il nome del file alla lista di parametri.
      push!(parameter_lists, merge(Dict("filename" => f), JSON.parse(s)))
    end
  end
  cd(prev_dir)
  return parameter_lists
end

function allequal(a)
  return all(x -> x == first(a), a)
end

# Costruzione della lista di istanti di tempo per la simulazione
# ==============================================================
function construct_step_list(parameters)
  # Restituisce la lista degli istanti di tempo attraversati dalla
  # simulazione. Se "simulation_time_step" è un multiplo (intero) di
  # "simulation_end_time", la lista comprenderà anche l'estremo finale.
  τ = parameters["simulation_time_step"]
  end_time = parameters["simulation_end_time"]
  return collect(range(0, end_time; step=τ))
end

# Calcolo di osservabili durante la simulazione
# =============================================
# Per calcolare gli autostati dell'operatore numero: calcolo anche la somma
# di tutti i coefficienti (così da verificare che sia pari a 1).
function levels(projs::Vector{MPO}, state::MPS)
  lev = [real(inner(state, p * state)) for p in projs]
  return [lev; sum(lev)]
end
function levels(projs::Vector{MPS}, state::MPS)
  lev = [real(inner(p, state)) for p in projs]
  return [lev; sum(lev)]
end

# Rilevazione della dimensione dei legami tra i siti degli MPS o MPO
function linkdims(m::Union{MPS, MPO})
  return [ITensors.dim(linkind(m, j)) for j ∈ 1:length(m)-1]
end

# Composizione di MPS e MPO
# =========================
function chain(left::MPS, right::MPS)
  # Questa funzione dovrebbe essere l'analogo di `chain` di mpnum: prende
  # due MPS e ne crea uno che ne è la concatenazione.
  # Per scriverla mi sono "ispirato" al codice della funzione `MPS` nel
  # file mps.jl, riga 308.

  midN = length(left) # È l'indice a cui si troverà il "buco"
  # Innanzitutto, riscalo i tag dei Link negli MPS di argomento, in
  # modo che la numerazione alla fine risulti corretta.
  # Facendo un po' di prove ho visto che la numerazione degli Index di
  # tipo Link nei MPS segue uno schema proprio, cioè non è legata a come
  # vengono chiamati i siti: va semplicemente da 1 alla lunghezza-1.
  for j in eachindex(left)
    replacetags!(left[j], "l=$j", "l=$j"; tags="Link")
    replacetags!(left[j], "l=$(j-1)", "l=$(j-1)"; tags="Link")
  end
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  # Spacchetto i tensori in `left` e `right` e creo un MPS con le
  # liste concatenate.
  M = MPS([left..., right...])
  # I siti "di confine", l'ultimo a dx di `left` e il primo a sx di
  # `right`, non sono ancora collegati. Creo un indice banale, di
  # dimensione 1, di tipo Link e lo aggiungo a quei due siti.
  # L'indice è _sempre_ banale perché per costruzione il MPS risultato
  # è il prodotto tensore dei due argomenti: non c'è interazione tra loro.
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = state(dag(missing_link), 1) * M[midN+1]

  return M
end

function chain(left::MPO, right::MPO)
  # Come sopra, ma per due MPO.
  midN = length(left) # È l'indice a cui si troverà il "buco"
  #for j in eachindex(left)
  #  replacetags!(left[j], "l=$j", "l=$j"; tags="Link")
  #  replacetags!(left[j], "l=$(j-1)", "l=$(j-1)"; tags="Link")
  #end
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  M = MPO([left..., right...])
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = M[midN+1] * state(dag(missing_link), 1)
  # L'ordine degli Index negli ITensor M[midN] e M[midN+1] non è quello
  # che si otterrebbe costruendo normalmente un MPO che si estende su
  # tutto l'array dei siti; per la precisione i due Index di tipo "Link"
  # sono scambiati.
  # Siccome però l'ordine degli Index posseduti da un ITensor non è
  # importante, conta solo quali indici ha, il risultato non dovrebbe
  # cambiare.
  return M
end

# Variante a numero di argomenti libero
chain(a::MPS, b...) = chain(a, chain(b...))
chain(a::MPO, b...) = chain(a, chain(b...))

function embed_slice(sites::Array{Index{Int64}},
                     range::UnitRange{Int},
                     slice::MPO)
  #=
  Prende un MPO definito solo su una fetta dei siti e lo estende con operatori
  identità per creare un MPO definito su tutti i siti.
  È richiesto che per i SiteType degli elementi di sites non compresi nella
  fetta data sia definito un operatore di nome "Id".

  Argomenti
  ---------
  · `sites::Array{Index{Int64}}`: l'array di siti dell'intero sistema.

  · `range::UnitRange{Int}`: un intervallo che indica il sito iniziale e il
    sito finale del MPO fornito che deve essere esteso.

  · `slice::Array{ITensors}`: il MPO da estendere.
  =#
  # Controllo dei parametri
  if length(slice) != length(range)
    throw(DimensionMismatch("Le dimensioni di slice e range non combaciano."))
  end
  if !issubset(range, eachindex(sites))
    throw(BoundsError(range, sites))
  end

  if range[begin] == 1 && range[end] == length(sites)
    mpo = slice
  elseif range[begin] == 1
    mpo = chain(slice,
                MPO(sites[range[end]+1 : end], "Id"))
  elseif range[end] == length(sites)
    mpo = chain(MPO(sites[1 : range[begin]-1], "Id"),
                slice)
  else
    mpo = chain(MPO(sites[1 : range[begin]-1], "Id"),
                slice,
                MPO(sites[range[end]+1 : end], "Id"))
  end
  return mpo
end

function embed_slice(sites::Array{Index{Int64}},
                     range::UnitRange{Int},
                     slice::MPS)
  #=
  Lo stesso di `embed_slice` per gli MPO, ma per gli MPS (e con "vecId" al
  posto di "Id"). Utile per gli stati vettorizzati che rappresentano
  osservabili da misurare.

  Argomenti
  ---------
  · `sites::Array{Index{Int64}}`: l'array di siti dell'intero sistema.

  · `range::UnitRange{Int}`: un intervallo che indica il sito iniziale e il
    sito finale del MPO fornito che deve essere esteso.

  · `slice::Array{ITensors}`: il MPS da estendere.
  =#
  # Controllo dei parametri
  if length(slice) != length(range)
    throw(DimensionMismatch("Le dimensioni di slice e range non combaciano."))
  end
  if !issubset(range, eachindex(sites))
    throw(BoundsError(range, sites))
  end

  if range[begin] == 1 && range[end] == length(sites)
    mpo = slice
  elseif range[begin] == 1
    mpo = chain(slice,
                MPS(sites[range[end]+1 : end], "vecId"))
  elseif range[end] == length(sites)
    mpo = chain(MPS(sites[1 : range[begin]-1], "vecId"),
                slice)
  else
    mpo = chain(MPS(sites[1 : range[begin]-1], "vecId"),
                slice,
                MPS(sites[range[end]+1 : end], "vecId"))
  end
  return mpo
end
