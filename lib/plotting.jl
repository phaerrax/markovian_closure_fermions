using Measures
using Base.Filesystem
using Plots
using JSON

# Grafici
# =======
function categorise_parameters(parameter_lists)
  # Analizzo l'array dei parametri e individuo quali parametri variano tra un
  # caso e l'altro, suddividendoli tra "distinti" se almeno uno dei parametri
  # è diverso tra le simulazioni, e "ripetuti" se sono invece tutti uguali.
  distinct = String[]

  # Devo controllare ciascuna lista di parametri per controllare se specifica
  # le dimensioni degli oscillatori separatamente per quelli caldi e freddi
  # o se dà solo una dimensione per entrambi.
  # Nel secondo caso, modifico il dizionario per riportarlo al primo caso.
  for dict in parameter_lists
    if (haskey(dict, "oscillator_space_dimension") &&
        !haskey(dict, "hot_oscillator_space_dimension") &&
        !haskey(dict, "cold_oscillator_space_dimension"))
      hotoscdim = pop!(dict, "oscillator_space_dimension")
      coldoscdim = hotoscdim
    end
  end

  for key in keys(parameter_lists[begin])
    test_list = [p[key] for p in parameter_lists]
    if !allequal(test_list)
      push!(distinct, key)
    end
  end
  repeated = setdiff(keys(parameter_lists[begin]), distinct)
  return distinct, repeated
end

function subplot_title(values_dict, keys)
  # Questa funzione costruisce il titolo personalizzato per ciascun sottografico,
  # che indica solo i parametri che cambiano da una simulazione all'altra
  #
  rootdirname = "simulazioni_tesi"
  sourcepath = Base.source_path()
  ind = findfirst(rootdirname, sourcepath)
  rootpath = sourcepath[begin:ind[end]]
  libpath = joinpath(rootpath, "lib")
  f = open(joinpath(libpath, "short_names_dictionary.json"), "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  hidden_parameters = ["skip_steps", "filename"]
  return join([short_name[k] * "=" * string(values_dict[k])
               for k in setdiff(keys, hidden_parameters)],
              ", ")
end

function shared_title_fake_plot(subject::String, parameters)
  #= Siccome Plots.jl non ha ancora, che io sappia, un modo per dare un titolo
     a un gruppo di grafici, mi arrangio con questa soluzione trovata su
     StackOverflow, che consiste nel creare un grafico vuoto che contiene il
     titolo voluto come annotazione.
     Il titolone contiene i parametri comuni a tutte le simulazioni (perciò
     posso prendere senza problemi uno degli elementi di parameter_lists
     qualunque) e lo uso come titolo del gruppo di grafici.
  =#
  # Inserire in questo array i parametri che non si vuole che appaiano nel
  # titolo:
  hidden_parameters = ["simulation_end_time", "skip_steps", "filename"]
  _, repeated_parameters = categorise_parameters(parameters)
  #
  rootdirname = "simulazioni_tesi"
  sourcepath = Base.source_path()
  ind = findfirst(rootdirname, sourcepath)
  rootpath = sourcepath[begin:ind[end]]
  libpath = joinpath(rootpath, "lib")
  f = open(joinpath(libpath, "short_names_dictionary.json"), "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  #
  shared_title = join([short_name[k] * "=" * string(parameters[begin][k])
                       for k in setdiff(repeated_parameters, hidden_parameters)],
                      ", ")
  y = ones(3) # Dati falsi per far apparire un grafico
  title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], text(subject * "\n" * shared_title)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
  return title_fake_plot
end

function groupplot(x_super,
    y_super,
    parameter_super;
    maxyrange = nothing,
    labels,
    linestyles,
    commonxlabel,
    commonylabel,
    plottitle,
    plotsize)
  #= Crea un'immagine che raggruppa i grafici dei dati delle coppie
     (X,Y) ∈ x_super × y_super; in alto viene posto un titolo, grande, che
     elenca anche i parametri usati nelle simulazioni e comuni a tutti i
     grafici; ad ogni grafico invece viene assegnato un titolo che contiene
     i parametri diversi tra una simulazione e l'altra.

     Argomenti
     ---------
     · `x_super`: un array, ogni elemento del quale è una lista di numeri
       che rappresentano le ascisse del grafico, ad esempio istanti di tempo,
       numeri dei siti, eccetera.

     · `y_super`: un array, ogni elemento del quale è quello che si passerebbe
       alla funzione Plots.plot insieme al rispettivo elemento Xᵢ ∈ x_super
       per disegnarne il grafico. Questo significa che ogni Yᵢ ∈ y_super è una 
       matrice M×N con M = length(Xᵢ); N è il numero di colonne, e ogni
       colonna rappresenta una serie di dati da graficare. Ad esempio, per
       graficare tutti assieme i numeri di occupazione di 10 siti si può
       creare una matrice di 10 colonne: ogni riga della matrice conterrà
       i numeri di occupazione dei siti a un certo istante di tempo.

     · `parameter_lists`: un array di dizionari, ciascuno contenente i
       parametri che definiscono la simulazione dei dati degli elementi di
       x_super e y_super.

     · `labels`::Matrix{String} oppure Vector{Matrix{String}}: un array di
       che contiene le etichette da assegnare alla linea di ciascuna quantità
       da disegnare.

     · `linestyles`: come `labels`, ma per gli stili delle linee.

     · `commonxlabel`: etichetta delle ascisse (comune a tutti)

     · `commonylabel`: etichetta delle ordinate (comune a tutti)
     
     · `maxyrange`: minimo e massimo valore delle ordinate da mostrare (per tutti) 

     · `plottitle`: titolo grande del grafico

     · `plotsize`: una Pair che indica la dimensione dei singoli grafici
  =#
  # Per poter meglio confrontare a vista i dati, imposto una scala delle
  # ordinate uguale per tutti i grafici.
  yminima = minimum.(y_super)
  ymaxima = maximum.(y_super)
  ylimits = (minimum(yminima), maximum(ymaxima))
  # Limito l'asse y a maxyrange _solo_ se i grafici non ci stanno già
  # dentro da soli.
  if maxyrange != nothing
    if ylimits[begin] < maxyrange[begin]
      ylimits = (maxyrange[begin], ylimits[end])
    end
    if maxyrange[end] < ylimits[end]
      ylimits = (ylimits[begin], maxyrange[end])
    end
  end

  # Smisto i parametri in ripetuti e non, per creare i titoli dei grafici.
  distinct_parameters, _ = categorise_parameters(parameter_super)
  # Calcolo la grandezza totale dell'immagine a partire da quella dei grafici.
  figuresize = (2, Int(ceil(length(x_super)/2))+0.5) .* plotsize
  # Se `labels` è un vettore di vettori riga di stringhe, significa che
  # ogni sottografico ha già il suo insieme di etichette: sono a posto.
  # Se invece `labels` è solo un vettore riga di stringhe, significa che
  # quelle etichette sono da usare per tutti i grafici: allora creo in
  # questo momento il vettore di vettori riga di stringhe ripetendo quello
  # fornito come argomento.
  if labels isa Matrix{String}
    newlabels = repeat([labels], length(parameter_super))
    labels = newlabels
  end
  # Ripeto lo stesso trattamento per `linestyles`.
  if linestyles isa Matrix{Symbol}
    newlinestyles = repeat([linestyles], length(parameter_super))
    linestyles = newlinestyles
  end

  # Creo i singoli grafici.
  subplots = [plot(X,
                   Y,
                   ylim=ylimits,
                   label=lab,
                   linestyle=lst,
                   legend=:outerright,
                   xlabel=commonxlabel,
                   ylabel=commonylabel,
                   title=subplot_title(p, distinct_parameters),
                   left_margin=5mm,
                   bottom_margin=5mm,
                   size=figuresize)
              for (X, Y, lab, lst, p) in zip(x_super,
                                             y_super,
                                             labels,
                                             linestyles,
                                             parameter_super)]
  # I grafici saranno disposti in una griglia con due colonne; se ho un numero
  # dispari di grafici, ne creo uno vuoto in modo da riempire il buco che
  # si crea (altrimenti mi becco un errore).
  if isodd(length(subplots))
    fakeplot = Plots.scatter(ones(2),
                             marker=0,
                             markeralpha=0,
                             ticks=nothing,
                             axis=false,
                             grid=false,
                             leg=false,
                             size=figuresize)
    push!(subplots, fakeplot)
  end
  # Creo il grafico che raggruppa tutto, insieme al titolo principale.
  group = Plots.plot(shared_title_fake_plot(plottitle, parameter_super),
                     Plots.plot(subplots..., layout=(length(subplots)÷2, 2)),
                     # Usa ÷ e non /, in modo da ottenere un Int!
                     layout=grid(2, 1, heights=[0.1, 0.9]))
  return group
end

function unifiedplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)
  #= Crea un grafico delle coppie di serie di dati (X,Y) ∈ x_super × y_super,
     tutte assieme.
     In alto viene posto un titolo, grande, che elenca anche i parametri usati
     nelle simulazioni e comuni a tutti i grafici; le etichette di ciascuna
     serie del grafico contengono invece i parametri che differiscono tra una
     simulazione e l'altra.

     Argomenti
     ---------
     · `x_super`: un array ogni elemento del quale è una lista di ascisse.

     · `y_super`: un array, ogni elemento del quale è una lista di ordinate
       associate a `x` da rappresentare nel grafico.

     · `parameter_lists`: un array di dizionari, ciascuno contenente i
       parametri che definiscono la simulazione dei dati degli elementi di
       ciascun elemento di y_super.

     · `linestyles`: come `labels`, ma per gli stili delle linee.

     · `xlabel`: etichetta delle ascisse

     · `ylabel`: etichetta delle ordinate

     · `plottitle`: titolo del grafico

     · `plotsize`: una Pair che indica la dimensione del grafico
  =#
  # Smisto i parametri in ripetuti e non, per creare le etichette dei grafici.
  distinct_parameters, _ = categorise_parameters(parameter_super)
  # Imposto la dimensione della figura (con un po' di spazio per il titolo).
  figuresize = (1, 1.25) .* plotsize
  plt = Plots.plot()
  for (X, Y, p) in zip(x_super, y_super, parameter_super)
    plt = Plots.plot!(X,
                      Y,
                      label=subplot_title(p, distinct_parameters),
                      linestyle=linestyle,
                      legend=:outerbottom,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      title=plottitle,
                      size=figuresize)
  end
  return plt
end

function unifiedlogplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)
  #= Crea un grafico delle coppie di serie di dati (X,Y) ∈ x_super × y_super,
     tutte assieme, con asse y logaritmico
     In alto viene posto un titolo, grande, che elenca anche i parametri usati
     nelle simulazioni e comuni a tutti i grafici; le etichette di ciascuna
     serie del grafico contengono invece i parametri che differiscono tra una
     simulazione e l'altra.
     In pratica, la funzione applica il logaritmo ai dati sull'asse delle
     ordinate e poi passa tutto a `unifiedplot`, in modo da non duplicare
     il codice.
     Non controllo che le ordinate siano positive: lascio l'onere di
     controllare che il grafico logaritmico si possa effettivamente fare
     a chi chiama la funzione.

     Argomenti
     ---------
     · `x_super`: un array ogni elemento del quale è una lista di ascisse.

     · `y_super`: un array, ogni elemento del quale è una lista di ordinate
       associate a `x` da rappresentare nel grafico.

     · `parameter_lists`: un array di dizionari, ciascuno contenente i
       parametri che definiscono la simulazione dei dati degli elementi di
       ciascun elemento di y_super.

     · `linestyles`: come `labels`, ma per gli stili delle linee.

     · `xlabel`: etichetta delle ascisse

     · `ylabel`: etichetta delle ordinate

     · `plottitle`: titolo del grafico

     · `plotsize`: una Pair che indica la dimensione del grafico
  =#
  return unifiedplot(x_super,
                     [log.(Y) for Y ∈ y_super],
                     parameter_super;
                     linestyle=linestyle,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     plottitle=plottitle,
                     plotsize=plotsize)
end
