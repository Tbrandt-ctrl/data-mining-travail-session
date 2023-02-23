library(caret)
library(lightgbm)

# Load the dataset

# HELLO 
test <- "test data"
data(iris)
iris$Species <- as.factor(iris$Species)

# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]

# Define the hyperparameters to tune and their search space
hyperparameters <- list(
  objective = "multiclass",
  metric = "multi_logloss",
  learning_rate = c(0.01, 0.05, 0.1),
  max_depth = c(3, 5, 7),
  num_leaves = c(20, 30, 40)
)

hyperparameters <- unique(hyperparameters)

# Create a training control object
train_control <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

AlgoritmListReg<-c("rf", "lm", "knn", "gbm")


# Train the model using the lightgbm package
model <- train(
  Species ~ .,
  data = train,
  methodList = AlgoritmListReg,
  tuneGrid = hyperparameters,
  trControl = train_control,
  metric = "Accuracy"
)

# Print the results
summary(model)
