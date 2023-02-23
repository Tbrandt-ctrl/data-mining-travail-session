library(ranger)

# Load the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

train$difflevel=factor(train$difflevel)
train$country=factor(train$country)

test$difflevel=factor(test$difflevel)
test$country=factor(test$country)

train$y <- as.factor(train$y)

# Split the train data into training and validation sets
set.seed(123) # for reproducibility


# Basic rf
basic_rf <- ranger(y ~., data=train, num.trees=500, importance = "impurity")
basic_rf_prediction <- predict(basic_rf, data=test)

# Plus d'arbres 
plusarbre_rf <- ranger(y ~., data=train, num.trees=1500, importance = "impurity")
plusarbre_rf_prediction <- predict(plusarbre_rf, data=test)
plusarbre_rf_prediction

# Plus d'arbre et probability TRUE

plusarbrepred_rf <- ranger(y ~., data=train, num.trees=1500, importance = "impurity", probability = TRUE, respect.unordered.factors = "partition")
plusarbrepred_rf_prediction <- predict(plusarbrepred_rf, data=test)


# ... et mtry 13 (minimise la prediction error)

rf_mtry_13 <- ranger(y ~., data=train, num.trees=1500, importance = "impurity",mtry=13,  probability = TRUE, respect.unordered.factors = "partition")
rf_mtry_13_pred <- predict(rf_mtry_13, data=test)$predictions


# Create a data frame for the prediction
basic_rf_prediction_df <- data.frame( y=basic_rf_prediction$predictions)

# Create the prediction file
write.csv(basic_rf_prediction_df, file = "predictions.csv")

# VIMP

# Test Mtry Loop

ncol(train)

best_mtry = floor(sqrt(ncol(train)))
lowest_brier = 1


for(test_mtry in 1:ncol(train)){
  rf_test_lopp <- ranger(y ~., data=train, num.trees=1500, importance = "impurity",mtry=test_mtry,  probability = TRUE, respect.unordered.factors = "partition")
  brier <- rf_test_lopp$prediction.error
  print(test_mtry)
  print(brier)
  
  if (brier < lowest_brier) {
    lowest_brier <- brier
    best_mtry <- test_mtry
  }
  
}

print(best_mtry) # 13




