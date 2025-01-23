using JSON, Tables, HDF5, CSV, ArgParse
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

function spinchain(::SiteType"vFermion", c::ModeChain)
    ℓ = OpSum()
    for (n, f) in zip(c.range, c.frequencies)
        ℓ += f * gkslcommutator("N", n)
    end
    for (n1, n2, g) in zip(c.range[1:(end - 1)], c.range[2:end], c.couplings)
        # Reorder `n1` and `n2`, otherwise `jwstring` doesn't put in the string factors
        # if `n1 > n2`. The exchange interaction operator is symmetric in `n1` and `n2`
        # so this reordering doesn't affect the result.
        n1, n2 = minmax(n1, n2)
        jws = jwstring(; start=n1, stop=n2)
        ℓ +=
            g * (
                gkslcommutator("A†", n1, jws..., "A", n2) +
                gkslcommutator("A", n1, jws..., "A†", n2)
            )
    end
    return ℓ
end

"""
    join(c1::ModeChain, c2::ModeChain, c1c2coupling)

Return a new `ModeChain` made by joining `c1` and `c2`, with `c1c2coupling` as the coupling
constant between the last site of `c1` and the first site of `c2`.
The ranges of the two chains do not have to be adjacent, but they cannot overlap.

# Example

```julia-repl
julia> c1 = ModeChain(1:3, fill(1,3), fill(0.5, 2));

julia> c2 = ModeChain(6:9, fill(2,4), fill(4,3));

julia> join(c1, c2, 0.25)
ModeChain([1, 2, 3, 6, 7, 8, 9], [1, 1, 1, 2, 2, 2, 2], [0.5, 0.5, 0.25, 4.0, 4.0, 4.0])
```
"""
function Base.join(c1::ModeChain, c2::ModeChain, c1c2coupling)
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
        error("Logic error")
    end
end

Base.length(c::ModeChain) = length(c.range)
Base.iterate(c::ModeChain) = iterate(c.range)
Base.iterate(c::ModeChain, i::Int) = iterate(c.range, i)

"""
    first(c::ModeChain, n::Integer)

Get the the mode chain `c` truncated to its first `n` elements.
"""
Base.first(c::ModeChain, n::Integer) =
    ModeChain(first(c.range, n), first(c.frequencies, n), first(c.couplings, n - 1))

"""
    last(c::ModeChain, n::Integer)

Get the the mode chain `c` truncated to its last `n` elements.
"""
Base.last(c::ModeChain, n::Integer) =
    ModeChain(last(c.range, n), last(c.frequencies, n), last(c.couplings, n - 1))

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

    truncated_envchain = first(chain, nenvironment)
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

"""
    parsecommandline(args...)

Set up a command-line argument parser for the program this function is included in.
The following arguments are always included:

```
  --input_parameters, -i
                        path to file with JSON dictionary of input parameters
  --system_sites, --ns
                        number of system sites
  --system_energy, --sysen
                        energy of the system's excited state (for single-site systems)
  --system_initial_state
                        initial state of the system (an ITensor StateName)
  --environment_chain_coefficients
                        path to file with chain coefficients of the environment(s)
  --environment_sites, --ne
                        number of environment sites
  --closure_sites, --nc
                        number of pseudomodes in the Markovian closure(s)
  --time_step, --dt
                        time step of the evolution
  --max_time, --maxt
                        total physical time of the evolution
  --max_bond_dimension, --bdim
                        maximum bond dimension of the state MPS
  --output, -o
                        path (basename) to output files
```

The JSON dictionary provided under the `input_parameters` is loaded first, and the other
command-line arguments afterwards. This means that if a parameter is given both in the JSON
file and through the command-line then the latter _overrides_ the former.

Other arguments can be added as desired, following the same syntax as
[`ArgParse.add_arg_table!`](@ref).
"""
function parsecommandline(args...)
    s = ArgParseSettings()
    add_arg_table!(
        s,
        ["--input_parameters", "-i"],
        Dict(
            :help => "path to file with JSON dictionary of input parameters",
            :arg_type => String,
        ),
        # open "core" system features
        ["--system_sites", "--ns"],
        Dict(:help => "number of system sites", :arg_type => Int),
        ["--system_energy", "--sysen"],
        Dict(
            :help => "energy of the system's excited state (for single-site systems)",
            :arg_type => Float64,
        ),
        ["--system_initial_state"],
        Dict(
            :help => "initial state of the system (an ITensor StateName)",
            :arg_type => String,
        ),
        ["--environment_chain_coefficients"],
        Dict(
            :help => "path to file with chain coefficients of the environment(s)",
            :arg_type => String,
        ),
        ["--environment_sites", "--ne"],
        Dict(:help => "number of environment sites", :arg_type => Int),
        ["--closure_sites", "--nc"],
        Dict(:help => "number of environment sites", :arg_type => Int),
        ["--time_step", "--dt"],
        Dict(:help => "time step of the evolution", :arg_type => Float64),
        ["--max_time", "--maxt"],
        Dict(:help => "total physical time of the evolution", :arg_type => Float64),
        ["--max_bond_dimension", "--bdim"],
        Dict(:help => "maximum bond dimension of the state MPS", :arg_type => Int),
        ["--observables", "--obs"],
        Dict(:help => "observables to be measured", :arg_type => String),
        # ↖ There are two ways the caller can specify the observables: either as a
        # dictionary or as a string (see the two `parseoperators` methods below).
        # Dictionaries are the "legacy" input used in JSON files, while strings are a new
        # way of inputting observables that suits the command-line interface well (I don't
        # know if ArgParse even accepts dictionaries as types of command-line at all).
        # The input routine, as far as the observables are concerned, works as follows.
        # 1) The JSON file, if provided, is loaded, and if the `observables` entry is
        #   present then the observable are taken from there. They can be a dictionary or
        #   a string, it doesn't matter, the `parseoperators` will distinguish them via
        #   the multiple dispatch functionality. Here ArgParse is not involved in loading
        #   the observables.
        # 2) The command-line arguments are read, and if `observables` is there, it either
        #   overrides the already existing observables or creates new ones. This time the
        #   corresponding argument can only be a string. This is why we can restrict the
        #   `arg_type` of the argument as String. At this stage we're not accepting
        #   dictionaries anymore.
        ["--output", "-o"],
        Dict(:help => "basename to output files", :arg_type => String),
    )
    add_arg_table!(s, args...) # additional user-specified CLI arguments

    # Load the input JSON file, if present.
    parsedargs_raw = parse_args(s)
    parsedargs =
        if !haskey(parsedargs_raw, "input_parameters") ||
            !isnothing(parsedargs_raw["input_parameters"])
            # ↖ Once the key is defined in `add_arg_table!`, the dictionary returned by
            # `parse_args` will always contain it, possibly with value `nothing` if the
            # argument is not given by the script caller. So check first that the key
            # doesn't exist in the dictionary (it most likely always does, but it's just to
            # be sure), and then also check whether its value is `nothing`.
            inputfile = pop!(parsedargs_raw, "input_parameters")
            @info "Reading parameters from $inputfile and from the command line"
            load_pars(inputfile)
        else
            @info "Reading parameters from the command line"
            Dict()
        end

    # Read arguments from the command line, overwriting existing ones.
    for (k, v) in parse_args(s)
        isnothing(v) || push!(parsedargs, k => v)
    end
    return parsedargs
end

function _expandsequence(seq)
    re = match(r"(?<name>\w+)\((?<sites>.+)\)", seq)
    sites = parse.(Int, split(re["sites"], ","))
    return [string(re["name"], "(", j, ")") for j in sites]
end

"""
    parseoperators(s::AbstractString)

Parse the string `s` as a list of `LocalOperator` objects, with the following rules:

- operators are written as products of local operators of the form `name(site)` where
    `name` is a string and `site` is an integer;
- operators on different sites can be multiplied together by writing them one after the
    other, i.e. `a(1)b(2)`;
- for convenience, writing a comma-separated list of numbers in the parentheses expands to a
    list of operators with the same name on each of the sites in the list, i.e. `a(1,2,3)`
    is interpreted as `a(1),a(2),a(3)`.

# Example

```julia-repl
julia> parseoperators("x(1)y(3),y(4),z(1,2,3)")
5-element Vector{LocalOperator}:
 x{1}y{3}
 y{4}
 z{1}
 z{2}
 z{3}
```
"""
function parseoperators(s::AbstractString)
    s *= ","  # add extra delimiter at the end (needed for regex below)
    # Split each occurrence made by anything between a word character and "),",
    # matching as few characters as possible between them.
    opstrings = chop.([r.match for r in eachmatch(r"\w+\(.+?\),", s * ",")])
    ops = LocalOperator[]
    i = 1
    while i <= length(opstrings)
        # Replace each sequence, i.e. an item like x(1,2,4), by its expansion.
        if contains(opstrings[i], ",")  # it is a sequence
            # ↖ Replace `opstrings[i]` with the expanded sequence, shifting the following
            # elements in the array to the right to make space for it. Then, restart the
            # loop iteration from the same index, which will be the first operator of the
            # just expanded sequence.
            splice!(opstrings, i:i, _expandsequence(opstrings[i]))
            continue
        end
        # Decide whether we have a product of operators, such as x(1)y(2), or a
        # sequence such as x(1,2,3)
        d = Dict()
        foreach(
            re -> push!(d, parse(Int, re["site"]) => re["name"]),
            eachmatch(r"(?<name>\w+?)\((?<site>\d+?)\)", opstrings[i]),
        )
        push!(ops, LocalOperator(d))
        i += 1
    end

    return ops
end

"""
    parseoperators(d::Dict)

Translate a `String => Vector{Int}` dictionary into a list of `LocalOperator` objects, where
each `(key, value)` pair is interpreted as the operator `key` on each of the sites contained
in `value`, separately.
This type of input does not support products of operators on different sites.

# Example

```julia-repl
julia> ops = Dict("x" => [1, 2, 3], "y" => [2]);

julia> parseoperators(ops)
4-element Vector{LocalOperator}:
 x{1}
 x{2}
 x{3}
 y{2}
```
"""
function parseoperators(d::Dict)
    ops = LocalOperator[]
    for (k, v) in d
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end
    return ops
end
