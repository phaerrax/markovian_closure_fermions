# a bunch of convenience functions that are currently helpful for testing
export load_pars, tfi_mpo, complex!, TFIgs, tfi_bondop

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

function tfi_mpo(J, h, sites)
    ampo = AutoMPO()
    for j in 1:(length(sites) - 1)
        add!(ampo, -J, "Sz", j, "Sz", j + 1)
        add!(ampo, -h, "Sx", j)
    end
    add!(ampo, -h, "Sx", length(sites))
    return MPO(ampo, sites)
end

function complex!(psi::MPS)
    #hack to make psi complex
    # I couldn't find a way to initialize a complex
    # valued tensor in ITensors
    for b in 1:length(psi)
        psi[b] *= (1.0 + 0.0im)
    end
    return psi
end

"get ground state of transverse-field Ising model"
function TFIgs(sites, h)
    ampo = AutoMPO()
    for j in 1:(length(sites) - 1)
        add!(ampo, -1.0, "Sz", j, "Sz", j + 1)
        add!(ampo, -h, "Sx", j)
    end
    add!(ampo, -h, "Sx", length(sites))
    H = MPO(ampo, sites)

    psi0 = randomMPS(sites)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 10, 20, 100, 100, 200)
    cutoff!(sweeps, 1E-10)
    energy, psi = dmrg(H, psi0, sweeps; quiet=true)
    return psi, energy
end

"create a BondOperator for the transverse-field ising model"
function tfi_bondop(sites, J, h)
    N = length(sites)
    H = BondOperator(sites)
    for b in 1:(N - 1)
        add!(H, -J, "Sz", "Sz", b)
        add!(H, -h, "Sx", b)
    end
    add!(H, -h, "Sx", N)
    return H
end

function measure!(psi::MPS, opname::String, i::Int)
    orthogonalize!(psi, i)
    return scalar(dag(psi[i]) * noprime(op(siteindex(psi, i), opname) * psi[i]))
end

measure!(psi::MPS, opname::String) = map(x -> measure!(psi, opname, x), 1:length(psi))
