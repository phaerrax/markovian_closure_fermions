#!/usr/bin/wolframscript
argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
cfsfile = argv[[1]];

Print["Reading coefficients from " <> cfsfile]

(*
    Read chain coefficients from files.
    The first row is the column name, and should be deleted.
    Moreover, the first (number) element of the couplings is the
    interaction coefficient between the system and the first site of the
    chain. It doesn't play any role here in the computation of the
    correlation function, so we discard it.
*)
coups = Delete[Table[item[[1]], {item, Import[cfsfile]}], 1]
freqs = Delete[Table[item[[2]], {item, Import[cfsfile]}], 1]
sysint = coups[[1]]
coups = Delete[coups, 1]
nsites = Length[freqs]

asympfrequency = Last[freqs]
asympcoupling = Last[coups]

(*
    Knowing the length of the chain and the approximate propagation speed
    of the excitations (which we take as the last coupling constant) we
    can estimate the time interval over which we won't see artifacts due
    to the finite size of the system:
        max_time = length / speed
*)
maxtime = 0.9 * nsites / (2 Last[coups])

(* Matrix of the TEDOPA chain in the single-excitation subspace *)
H = DiagonalMatrix[freqs, 0] + DiagonalMatrix[coups, 1] + DiagonalMatrix[coups, -1]

(*
    The correlation function of this TEDOPA chain is given by
        f(t) = (psi0, X_1(t) X_1(0) psi0)
    where X_1 = A_1 + Adag_1 and psi0 is the vacuum state. It simplifies to
        f(t) = (psi0, A_1(t) Adag_1 psi0).
    The vacuum psi0 is an eigenstate of the Hamiltonian, with
    eigenvalue 0, so
        f(t) = (psi0, A_1 exp(-itH) Adag psi0) =
             = (s1, exp(-itH) s1)
    where the s1 state, representing an excitation on the first site
    of the chain, is written as [1 0 … 0] in the single-excitation
    subspace. This also means that f(t) can be obtained as the first
    coefficient of
        exp(-itH) s1
    which we can compute faster by NDSolve:
        g'(t) + i H g(t) = 0,
        g(0) = s1
    and then f(t) = g(t)[[1]], instead of using MatrixExp directly.
*)

(* corrf[t_] := 1 / 2 * ConjugateTranspose[start].MatrixExp[-I * t * H, start] *)
start = Join[{1}, Table[0, nsites-1]]
sol = NDSolve[{y'[t] + I H.y[t] == 0, y[0] == sysint^2 * start}, y, {t, 0, maxtime}]
g[t_] := y[t] /. First[sol]
corrf[t_?NumericQ] := g[t][[1]]
(*
    Watch out: writing
        corrf[t] := g[t][[1]]
    only, for some mysterious, reason doesn't give the correct results.
    The function MUST be defined appending the _NumericQ? thing to the variable.
*)
corrfgfx = Plot[{Re[corrf[t]], Im[corrf[t]]}, {t, 0, maxtime}, PlotLegends -> {"Re", "Im"}, PlotRange -> Full];

(* Spectral density corresponding to the correlation function *)
sdf[x_?NumericQ, upperbound_:maxtime] := (1 / Pi) NIntegrate[
    Cos[x t] Re[corrf[t]] - Sin[x t] Im[corrf[t]],
    {t, 0, upperbound},
    AccuracyGoal -> 5,
    Method -> {"GlobalAdaptive", Method -> "GaussKronrodRule"}
]
(* We can use a more "economical" integration method, the integrand is regular enough *)
sdfgfx = Plot[sdf[w], {w, asympfrequency - 2.1 asympcoupling, asympfrequency + 2.1 asympcoupling}, PlotRange -> Full];

(* Expected solution: spectral density function of the semicircle *)
expsdf[x_] := Piecewise[
    {
        {0, x < asympfrequency - 2 asympcoupling},
        {0, x > asympfrequency + 2 asympcoupling}
    },
    1 / (2 Pi) Sqrt[(2 asympcoupling - asympfrequency + x)(2 asympcoupling + asympfrequency - x)]
]
expsdfgfx = Plot[expsdf[w], {w, asympfrequency - 2.1 asympcoupling, asympfrequency + 2.1 asympcoupling}, PlotRange -> Full];

(* Expected solution: correlation function of the semicircle *)
expcorrf[t_?NumericQ] := NIntegrate[
    E^(-I x t) expsdf[x],
    {x, asympfrequency - 2 asympcoupling, asympfrequency + 2 asympcoupling},
    AccuracyGoal -> 5,
    Method -> {"GlobalAdaptive", Method -> "GaussKronrodRule"}
]
expcorrfgfx = Plot[{Re[expcorrf[t]], Im[expcorrf[t]]}, {t, 0, maxtime}, PlotLegends -> {"Re", "Im"}, PlotRange -> Full];

exportfilename = cfsfile <> ".pdf"
Print["Exporting plots in " <> exportfilename]
Export[exportfilename, GraphicsGrid[{{corrfgfx, sdfgfx}, {expcorrfgfx, expsdfgfx}}]]
