using JSON, Tables, HDF5, CSV
using Statistics: mean

interleave(v...) = collect(Iterators.flatten(zip(v...)))

"""
    load_pars(file_name::String)

Load the JSON file `file_name` into a dictionary.
"""
function load_pars(file_name::String)
    input = open(file_name)
    s = read(input, String)
    # Aggiungo anche il nome del file alla lista di parametri.
    p = JSON.parse(s)
    return p
end

function enlargelinks_delta(v::MPS, new_d)
    @warn "This function currently doesn't seem to work when QNs are involved."
    N = length(v)
    if N == 1
        v_overlap = 1
        @debug "The length of the MPS is 1. There are no bonds to enlarge."
        return v, v_overlap
    end

    v_ext = copy(v)
    if hasqns(v[1])
        #=
           The `hasqns` part comes from the constructor method of an MPS with QNs.
           The QN index structure of an MPS is as follows:

                                      s[n]                        s[n+1]

                            │                            │
                            │      l[n]                  │     l[n+1]
                          ╭───╮    <In>                ╭───╮   <In>
           ───────────────│ n │───────╮ ╭──────────────│n+1│─────────
           dag(l[n-1])    ╰───╯       │ │ dag(l[n])    ╰───╯
           <Out>                      │ │ <Out>
                                      │ │
                            ╭─────────╯ ╰─────────╮
                            │                     │
                       ┌╶╶╶╶╶╶╶╶╶╶╶╶╶┐      ┌╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┐
                       ╎delta(       ╎      ╎delta(          ╎
                       ╎  dag(l[n]), ╎      ╎  l[n],         ╎
                       ╎  new_index  ╎      ╎  dag(new_index)╎
                       ╎)            ╎      ╎)               ╎
                       └╶╶╶╶╶╶╶╶╶╶╶╶╶┘      └╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶┘
                            │                     │

        =#
        # We need now to calculate the flux of the QNs in the MPS. Note that in the original
        # MPS constructor, where this code comes from, the flux is calculated from the
        # states that make up the product-state MPS, not from the MPS itself (that obviously
        # doesn't yet exist there). We need to recreate those states starting from the MPS
        # `v`: we cannot use just `v` because it contains link indices too, and they mess
        # up the calculation of the flux.
        # If the MPS is a product state, then we can recover the states by contracting the
        # link indices on each site: since they are one-dimensional, we contract them with
        # an ITensor made from `onehot(... => 1)`, and the link index disappears without
        # affecting the rest.
        if any(linkdims(v_ext) .!= 1)
            error(
                "MPS is not a triv_extial product state. Enlarging non-product-states is " *
                "currently not supported.",
            )
        end
        states = Vector{ITensor}(undef, N)
        states[1] = v_ext[1] * dag(onehot(linkind(v_ext, 1) => 1))
        for n in 2:(N - 1)
            states[n] =
                v_ext[n] *
                onehot(linkind(v_ext, n - 1) => 1) *
                onehot(dag(linkind(v_ext, n)) => 1)
        end
        states[N] = v_ext[N] * onehot(linkind(v_ext, N - 1) => 1)

        enlarged_links = Vector{ITensors.QNIndex}(undef, N - 1)
        lflux = sum(flux, states[1:(end - 1)])
        for bond in (N - 1):-1:1
            enlarged_links[bond] = dag(
                Index(lflux => new_d; tags=tags(commonind(v_ext[bond], v_ext[bond + 1])))
            )
            lflux -= flux(states[bond])
        end
    else
        enlarged_links = [
            Index(new_d; tags=tags(commonind(v_ext[bond], v_ext[bond + 1]))) for
            bond in 1:(N - 1)
        ]
    end

    for bond in 1:(N - 1)
        bond_index = commonind(v_ext[bond], v_ext[bond + 1])
        # Remember to `dag` some indices so that the directions of the QN match (see above)
        v_ext[bond] = v_ext[bond] * delta(dag(bond_index), enlarged_links[bond])
        v_ext[bond + 1] = v_ext[bond + 1] * delta(bond_index, dag(enlarged_links[bond]))
    end

    v_overlap = dot(v, v_ext)
    @debug "Overlap ⟨original|extended⟩: $v_overlap"
    return v_ext, v_overlap
end

nullfile() = "/dev/null"

struct ModeChain
    range
    frequencies
    couplings
    function ModeChain(input_range, input_frequencies, input_couplings)
        @assert allequal([
            length(input_range), length(input_frequencies), length(input_couplings) + 1
        ])
        return new(input_range, input_frequencies, input_couplings)
    end
end

function spinchain(::SiteType"Fermion", c::ModeChain)
    ad_h = OpSum()
    for (n, f) in zip(c.range, c.frequencies)
        ad_h += f, "n", n
    end
    for (n1, n2, g) in zip(c.range[1:(end - 1)], c.range[2:end], c.couplings)
        ad_h += g, "cdag", n1, "c", n2
        ad_h += g, "cdag", n2, "c", n1
    end
    return ad_h
end

function join(c1::ModeChain, c2::ModeChain, c1c2coupling)
    if first(c1.range) ≤ last(c2.range) && first(c2.range) ≤ last(c1.range)
        error("The ranges of the given ModeChains overlap.")
    elseif first(c1.range) < first(c2.range)  # find out which chain is on the left
        return ModeChain(
            [c1.range; c2.range],
            [c1.frequencies; c2.frequencies],
            [c1.couplings; c1c2coupling; c2.couplings],
        )
    elseif first(c2.range) < first(c1.range)
        return ModeChain(
            [c2.range; c1.range],
            [c2.frequencies; c1.frequencies],
            [c2.couplings; c1c2coupling; c1.couplings],
        )
    else
        error("?")
    end
end

Base.length(c::ModeChain) = length(c.range)
Base.iterate(c::ModeChain) = iterate(c.range)
Base.iterate(c::ModeChain, i::Int) = iterate(c.range, i)

"""
    markovianclosure(chain::ModeChain, nclosure, nenvironment)

Replace the ModeChain `chain` with a truncated chain of `nenvironment` elements plus a
Markovian closure made of `nclosure` pseudomodes.
The asymptotic frequency and coupling coefficients are determined automatically from the
average of the chain cofficients from `nenvironment+1` to the end, but they can also be
given manually with the keyword arguments `asymptoticfrequency` and `asymptoticcoupling`.
"""
function markovianclosure(
    chain::ModeChain,
    nclosure,
    nenvironment;
    asymptoticfrequency=nothing,
    asymptoticcoupling=nothing,
)
    if isnothing(asymptoticfrequency)
        asymptoticfrequency = mean(chain.frequencies[(nenvironment + 1):end])
    end
    if isnothing(asymptoticcoupling)
        asymptoticcoupling = mean(chain.couplings[(nenvironment + 1):end])
    end

    truncated_envchain = ModeChain(
        first(chain.range, nenvironment),
        first(chain.frequencies, nenvironment),
        first(chain.couplings, nenvironment - 1),
    )
    mc = markovianclosure_parameters(asymptoticfrequency, asymptoticcoupling, nclosure)

    return truncated_envchain, mc
end

"""
    pack!(outputfilename; argsdict, expvals_file, bonddimensions_file, walltime_file)

Gather all simulation input and output in a single HDF5 file `outputfilename`.
All input parameters used for the simulation, except for the name of the JSON input file
itself (if provided) will be in the output file together with the expectation values of the
selected observables, the bond dimensions of the evolved MPS and the wall-clock time spent
computing each step, for each step of the time evolution.

The `expvals_file`, `bonddimensions_file` and `walltime_file` files will be _deleted_ after
their contents are transferred to the HDF5 file.
"""
function pack!(outputfilename; argsdict, expvals_file, bonddimensions_file, walltime_file)
    # We remove the "observable" entry from the dictionary since we don't need it in the
    # output, it's already in the headers of the simulation results.
    # Same for "input_parameters", since its contents are what we're actually writing in
    # the HDF5 file.
    dict = deepcopy(argsdict)
    delete!(dict, "observables")
    delete!(dict, "input_parameters")
    delete!(dict, "name")

    h5open(outputfilename, "w") do hf
        for (k, v) in dict
            write(hf, k, v)
        end
        measurements = CSV.File(expvals_file)
        for col in propertynames(measurements)
            write(hf, string("simulation_results/", col), measurements[col])
        end

        # Pack the bond dimensions in a single Matrix, where the i-th columns is the link
        # between site i and i+1.
        bonddims = Matrix{Int}(Tables.matrix(CSV.File(bonddimensions_file; drop=[:time])))
        write(hf, "bond_dimensions", bonddims)

        # Read the simulation time per step in a Vector. There's only one column in the
        # file.
        walltime = CSV.File(walltime_file)
        write(hf, "simulation_wall_time", walltime[propertynames(walltime)[1]])
    end

    @info "Output written on $outputfilename"
    rm(expvals_file)
    rm(bonddimensions_file)
    rm(walltime_file)

    return nothing
end

function simulation_files_info(;
    measurements_file=nothing, bonddims_file=nothing, simtime_file=nothing
)
    str = "You can follow the time evolution step by step in the following files:"
    if !(isnothing(measurements_file) || measurements_file == nullfile())
        str *= "\n$measurements_file\t for the expectation values"
    end
    if !(isnothing(bonddims_file) || bonddims_file == nullfile())
        str *= "\n$bonddims_file\t for the bond dimensions of the evolved MPS"
    end
    if !(isnothing(simtime_file) || simtime_file == nullfile())
        str *= "\n$simtime_file\t for the wall-clock time spent computing each step"
    end
    @info str
end
