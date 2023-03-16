#!/usr/bin/wolframscript

argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
configfile = argv[[1]];

Print["Reading configuration parameters from " <> configfile]
Print[
    "Please note that this script works only when the original " <>
    "spectral density is a semicircle, transformed with TF-TEDOPA."
]

(*
    Read configuration coefficients from files.
    Importing a JSON file returns a "list of rules" from which we can
    obtain the values by using the substitution
        "key" /. Import[file]
*)
config = Import[configfile]
originaldomain = "domain" /. config
minfreq = originaldomain[[1]]
maxfreq = originaldomain[[2]]
pars = "parameters" /. config
originalfrequency = pars[[1]]
originalcoupling = pars[[2]]
chempot = "chemical_potential" /. config
temperature = "temperature" /. config
nsites = "number_of_oscillators" /. config

semicirclecorrf[x_, omega_, kappa_] := Piecewise[
    {
        {0, x < omega - 2 kappa},
        {0, x > omega + 2 kappa}
    },
    1 / (2 Pi) Sqrt[(2 kappa - omega + x)(2 kappa + omega - x)]
]

(*
    Read chain coefficients from files.
    The first row is the column name, and should be deleted.
    Moreover, the first (number) element of the couplings is the
    interaction coefficient between the system and the first site of the
    chain. It doesn't play any role here in the computation of the
    correlation function, so we discard it.
*)
upperchainfile = "upper_chain_file" /. config
lowerchainfile = "lower_chain_file" /. config
mixedchainfile = "mixed_chain_file" /. config

uppercoups = Take[
    Delete[Table[item[[1]], {item, Import[upperchainfile]}], 1],
    nsites
]
upperfreqs = Take[
    Delete[Table[item[[2]], {item, Import[upperchainfile]}], 1],
    nsites
]
uppersysint = uppercoups[[1]]
uppercoups = Delete[uppercoups, 1]

lowercoups = Take[
    Delete[Table[item[[1]], {item, Import[lowerchainfile]}], 1],
    nsites
]
lowerfreqs = Take[
    Delete[Table[item[[2]], {item, Import[lowerchainfile]}], 1],
    nsites
]
lowersysint = lowercoups[[1]]
lowercoups = Delete[lowercoups, 1]

mixedcoups = Take[
    Delete[Table[item[[1]], {item, Import[mixedchainfile]}], 1],
    nsites
]
mixedfreqs = Take[
    Delete[Table[item[[2]], {item, Import[mixedchainfile]}], 1],
    nsites
]
mixedsysint = mixedcoups[[1]]
mixedcoups = Delete[mixedcoups, 1]

(*
    Knowing the length of the chain and the approximate propagation speed
    of the excitations (which we take as the last coupling constant) we
    can estimate the time interval over which we won't see artifacts due
    to the finite size of the system:
        max_time = length / speed
*)
maxtime = 0.9 * nsites / (2 originalcoupling)
Print[maxtime]

(* Matrix of the TEDOPA chain in the single-excitation subspace *)
upperH = DiagonalMatrix[upperfreqs, 0] +
    DiagonalMatrix[uppercoups, 1] +
    DiagonalMatrix[uppercoups, -1];
lowerH = DiagonalMatrix[lowerfreqs, 0] +
    DiagonalMatrix[lowercoups, 1] +
    DiagonalMatrix[lowercoups, -1];
mixedH = DiagonalMatrix[mixedfreqs, 0] +
    DiagonalMatrix[mixedcoups, 1] +
    DiagonalMatrix[mixedcoups, -1];

(*
    The correlation function of this TEDOPA chain is given by
        f(t) = sysint^2 (psi0, X_1(t) X_1(0) psi0)
    where X_1 = A_1 + Adag_1 and psi0 is the vacuum state. It simplifies to
        f(t) = sysint^2 (psi0, A_1(t) Adag_1 psi0).
    The vacuum psi0 is an eigenstate of the Hamiltonian, with
    eigenvalue 0, so
        f(t) = sysint^2 (psi0, A_1 exp(-itH) Adag psi0) =
             = sysint^2 (s1, exp(-itH) s1)
    where the s1 state, representing an excitation on the first site
    of the chain, is written as [1 0 … 0] in the single-excitation
    subspace. This also means that f(t) can be obtained as the first
    coefficient of
        sysint^2 exp(-itH) s1
    which we can compute faster by NDSolve with the expression
        g'(t) + i H g(t) = 0,
        g(0) = sysint^2 s1
    and then f(t) = g(t)[[1]], instead of using MatrixExp.
*)
start = Join[{1}, Table[0, nsites-1]]

uppersol = NDSolve[
    {y'[t] + I upperH.y[t] == 0, y[0] == uppersysint^2 * start},
    y,
    {t, 0, maxtime}
]
uppercf[t_] := y[t] /. First[uppersol]

lowersol = NDSolve[
    {y'[t] + I lowerH.y[t] == 0, y[0] == lowersysint^2 * start},
    y,
    {t, 0, maxtime}
]
lowercf[t_] := y[t] /. First[lowersol]

separatecorrf[t_?NumericQ] := uppercf[t][[1]] + lowercf[t][[1]]

mixedsol = NDSolve[
    {y'[t] + I mixedH.y[t] == 0, y[0] == mixedsysint^2 * start},
    y,
    {t, 0, maxtime}
]
mixedcf[t_] := y[t] /. First[mixedsol]

mixedcorrf[t_?NumericQ] := mixedcf[t][[1]]
(*
    Watch out: writing
        corrf[t] := g[t][[1]]
    only, for some mysterious, reason doesn't give the correct results.
    The function MUST be defined appending the _NumericQ? thing to
    the variable.
*)
separatecorrfgfx[cutoff_:maxtime] := Plot[
    {Re[separatecorrf[t]], Im[separatecorrf[t]]},
    {t, 0, cutoff},
    PlotLegends -> {"Re", "Im"},
    PlotLabel -> "Separate chains correlation fn",
    PlotRange -> Full
];
mixedcorrfgfx[cutoff_:maxtime] := Plot[
    {Re[mixedcorrf[t]], Im[mixedcorrf[t]]},
    {t, 0, cutoff},
    PlotLegends -> {"Re", "Im"},
    PlotLabel -> "Mixed chain correlation fn",
    PlotRange -> Full
];

(*
    Spectral density corresponding to the correlation function:
    The inverse Fourier transform gives us a certain function j.
    We know this function is (should be) related to the original spectral
    density function J0 by
                / J0(mu-x) if 0 < x < mu,
        j(x) = <
                \ J0(mu+x) if mu < x < maxfreq.
    so we could try j(mu-x) or j(x-mu) and see if one of them sticks.
*)
sdf[x_?NumericQ, upperbound_:maxtime] := (1 / Pi) NIntegrate[
    Cos[x t] Re[corrf[t]] - Sin[x t] Im[corrf[t]],
    {t, 0, upperbound},
    AccuracyGoal -> 5,
    (*
        We can use a more "economical" integration method; the integrand
        is regular enough.
    *)
    Method -> {"GlobalAdaptive", Method -> "GaussKronrodRule"}
]
(* Expected solution: spectral density function of the semicircle *)
expsdf[x_] := semicirclecorrf[x, originalfrequency, originalcoupling]
expsdfgfx = Plot[
    expsdf[w],
    {w, minfreq, maxfreq},
    PlotLabel -> "Spectral density",
    PlotRange -> Full
];

(* Expected solution: correlation function of the semicircle *)
expcorrf[t_?NumericQ] := NIntegrate[
    E^(-I x t) Piecewise[
        {
            {expsdf[chempot - x] + expsdf[chempot + x], 0 < x < chempot},
            {expsdf[chempot + x], chempot < x < maxfreq - chempot}
        },
        0
    ],
    {x, 0, maxfreq},
    AccuracyGoal -> 5,
    Method -> {"GlobalAdaptive", Method -> "GaussKronrodRule"}
]
expcorrfgfx[cutoff_:maxtime] := Plot[
    {Re[expcorrf[t]], Im[expcorrf[t]]},
    {t, 0, cutoff},
    PlotLegends -> {"Re", "Im"},
    PlotLabel -> "Expected",
    PlotRange -> Full
];
diffmixedcorrfgfx[cutoff_:maxtime] := Plot[
    Abs[expcorrf[t]-mixedcorrf[t]],
    {t, 0, cutoff},
    PlotLabel -> "Difference expected/mixed chain",
    PlotRange -> Full
];
diffseparatecorrfgfx[cutoff_:maxtime] := Plot[
    Abs[expcorrf[t]-separatecorrf[t]],
    {t, 0, cutoff},
    PlotLabel -> "Difference expected/separate chains",
    PlotRange -> Full
];

exportfilename = configfile <> ".pdf"
Print["Exporting plots in " <> exportfilename]
Export[
    exportfilename,
    GraphicsGrid[
        {
            {expcorrfgfx[2], expsdfgfx},
            {mixedcorrfgfx[2], separatecorrfgfx[2]},
            {diffmixedcorrfgfx[2], diffseparatecorrfgfx[2]}
        }
    ]
]
