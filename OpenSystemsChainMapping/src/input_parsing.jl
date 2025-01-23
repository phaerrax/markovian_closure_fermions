"""
    load_pars(file_name::String)

Load the JSON file `file_name` into a dictionary.
"""
function load_pars(file_name::String)
    input = open(file_name)
    s = read(input, String)
    # Add the file name to the parameter list too.
    p = JSON.parse(s)
    return p
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
    opstrings = Base.chop.([r.match for r in eachmatch(r"\w+\(.+?\),", s * ",")])
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

Translate a dictionary into a list of `LocalOperator` objects, where each `(key, value)`
pair is interpreted as the operator `key` (a string) on each of the sites contained in
`value` (a list of integers), separately.
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
