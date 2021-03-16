library("reticulate")

print("Importing python module")
model <- import("model.graph_mcmc")

print("Instantiating graph")
graph <- model$Graph_MCMC(vector())

print("Reading from file")
graph$read_from_file("generated.gml")

print("Detecting partition")
graph$partition(2, 5)

print("Sampling from posterior")
marginals <- graph$mcmc(100, verbose = TRUE)
