push!(LOAD_PATH, "../src/")

using Documenter
using TimeEvoVecMPS

makedocs(
    sitename = "TimeEvoVecMPS",
    format = Documenter.HTML(),
    modules = [TimeEvoVecMPS],
    remotes = nothing,
    repo = Remotes.GitHub("phaerrax", "markovian_closure_fermions"),
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
