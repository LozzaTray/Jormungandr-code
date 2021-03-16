library("reticulate")

print("Importing python module...")
model <- import("model.graph_mcmc")

print("Instantiating graph...")
graph <- model$Graph_MCMC(vector())

print("Reading from file...")
graph$read_from_file("generated.gml")

print("Detecting partition...")
graph$partition(B_min = 2, B_max = 5)

print("Sampling from posterior...")
B <- graph$mcmc(100, verbose = TRUE)

print("Generating features...")
feature_names <- graph$get_feature_names()
X <- graph$generate_feature_matrix()
Y <- graph$generate_posterior()

cat("X dimension: ", dim(X), "\n")
cat("Y dimension: ", dim(Y), "\n")

D <- dim(X)[2]


library(keras)
# define model
print("Defining model...")
model <- keras_model_sequential()
model %>%
    layer_dense(
        units = B,
        input_shape = c(D),
        activation = "softmax",
        name = "BlockProbs"
    )


# compile model
model %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

# train
model %>% fit(X, Y, epochs = 10, verbose = 2)

# test metrics
score <- model %>% evaluate(X, Y, verbose = 0)

cat("Training loss:", score$loss, "\n")

# predict test set
predictions <- model %>% predict(X)