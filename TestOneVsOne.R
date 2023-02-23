# Load the required libraries
library(randomForest)

# Load the example dataset from the mlbench package
library(mlbench)
data(iris)

# Split the dataset into training and testing sets
set.seed(123)
train_idx <- sample(nrow(iris), round(0.7 * nrow(iris)))
train_data <- iris[train_idx, ]
test_data <- iris[-train_idx, ]

# Train multiple binary classifiers using the random forest algorithm
class_pairs <- combn(levels(iris$Species), 2, simplify = FALSE)
rf_models <- list()
for (i in seq_along(class_pairs)) {
  curr_pair <- class_pairs[[i]]
  curr_train_data <- train_data[train_data$Species %in% curr_pair, ]
  curr_train_data$Species <- factor(curr_train_data$Species)
  rf_models[[i]] <- randomForest(Species ~ ., data = curr_train_data, ntree = 100)
}

# Use the trained classifiers to predict the class of each test instance
test_preds <- matrix(0, nrow = nrow(test_data), ncol = length(class_pairs))
for (i in seq_along(class_pairs)) {
  curr_pair <- class_pairs[[i]]
  curr_test_data <- test_data[test_data$Species %in% curr_pair, ]
  curr_test_data$Species <- factor(curr_test_data$Species)
  curr_rf_model <- rf_models[[i]]
  curr_preds <- predict(curr_rf_model, newdata = curr_test_data)
  test_preds[test_data$Species %in% curr_pair, i] <- curr_preds
}

# Aggregate the predictions using the voting strategy
final_preds <- apply(test_preds, 1, function(x) {
  if (sum(x == "1") > sum(x == "2")) {
    return(class_pairs[[1]][1])
  } else {
    return(class_pairs[[1]][2])
  }
})

# Evaluate the accuracy of the final predictions
true_labels <- test_data$Species
accuracy <- mean(final_preds == true_labels)
print(paste0("Accuracy: ", round(accuracy, 3)))
