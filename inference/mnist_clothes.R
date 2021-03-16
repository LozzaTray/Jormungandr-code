# example from:
# https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification/

library(keras)
library(tfdatasets)
library(dplyr)

## % means binary operations

print("Downloading dataset...")
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

class_names <- c(
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
)

print("Dimensions: ")
print(dim(train_images))

library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

# scale down from 0-255 to 0-1
train_images <- train_images / 255
test_images <- test_images / 255

print("Plotting first 25 training images...")
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = "i", yaxs = "i")
for (i in 1:25) {
    img <- train_images[i, , ]
    img <- t(apply(img, 2, rev))
    image(1:28, 1:28, img,
        col = gray((0:255) / 255), xaxt = "n", yaxt = "n",
        main = paste(class_names[train_labels[i] + 1])
    )
}

# %foo% is R syntax for custom binary operators
# in dplyr package %>% is used as a pipe >> chains function calls

# define model
model <- keras_model_sequential()
model %>%
    layer_flatten(input_shape = c(28, 28)) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")

# compile model
model %>% compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
)

# train
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

# test metrics
score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat("Test loss:", score$loss, "\n")
cat("Test accuracy:", score$acc, "\n")

# predict test set
predictions <- model %>% predict(test_images)


# plot test examples
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = "i", yaxs = "i")
for (i in 1:25) {
    img <- test_images[i, , ]
    img <- t(apply(img, 2, rev))
    # subtract 1 as labels go from 0 to 9
    predicted_label <- which.max(predictions[i, ]) - 1
    true_label <- test_labels[i]
    if (predicted_label == true_label) {
        color <- "#008800"
    } else {
        color <- "#bb0000"
    }
    image(1:28, 1:28, img,
        col = gray((0:255) / 255), xaxt = "n", yaxt = "n",
        main = paste0(
            class_names[predicted_label + 1], " (",
            class_names[true_label + 1], ")"
        ),
        col.main = color
    )
}