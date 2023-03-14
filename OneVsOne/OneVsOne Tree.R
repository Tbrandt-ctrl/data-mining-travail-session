library(caret)
library(rpart)
library(rpart.plot)

set.seed(123)


### HELPING FUNCTIONS ###

# Standardisation 

stdf=function(dat1,dat2)
{
  # input:
  # dat1 = data frame à standardiser avec ses propres moyennes et SD -> Entraînement
  # dat2 = data frame à standardiser en utilisant les moyennes et SD du premier -> Validation
  # output = liste avec les 2 data frames standardisés
  mu=apply(dat1,2,mean)
  std=apply(dat1,2,sd)
  list(as.data.frame(scale(dat1, center=mu , scale=std)),
       as.data.frame(scale(dat2, center=mu , scale=std)))
}

### DATA PREPARATION ###

### PRÉPARATION DES DONNÉES ###

# Load the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Création de facteurs pour les variables difflevel, country et le y
train$difflevel=factor(train$difflevel)
train$country=factor(train$country)

test$difflevel=factor(test$difflevel)
test$country=factor(test$country)

# Noms des variables à traiter comme catégorielles
namescat=c("country","difflevel", "y", "id")

# Split the dataset into training and validation

train_index = createDataPartition(train$y, p = .80, list = F) # récupération aléatoire d'index

traintrain <- train[train_index,] # 4000 observations
trainvalid <- train[-train_index,] # 1000 observations

# Standardisation

outstd_comp=stdf(traintrain[,!(names(traintrain) %in% namescat)], # standardisation des sets de comparaison
                 trainvalid[,!(names(trainvalid) %in% namescat)])

outstd = stdf(train[,!(names(train) %in% namescat)], test[,!(names(test) %in% namescat)])

stdtraintrain = cbind(traintrain[, (names(traintrain) %in% namescat)], outstd_comp[1])
stdtrainvalid = cbind(trainvalid[, (names(trainvalid) %in% namescat)], outstd_comp[2])

stdtrain =cbind(train[,(names(train) %in% namescat)], outstd[1])
stdtest =cbind(test[,(names(test) %in% namescat)], outstd[2])

### MODEL TRAINING ### 

stdtraintrain$y <- as.factor(stdtraintrain$y)

class_pairs <- combn(levels(stdtraintrain$y), 2, simplify = FALSE)

## Modèles

tree_models_comp <- list()
tree_models_test <- list()

best_maxdepth <- list()
best_cp <- list()

for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
  
  # VALIDATION
  
  curr_stdtraintrain <- stdtraintrain[stdtraintrain$y %in% curr_pair, ]
  curr_stdtraintrain$y <- factor(curr_stdtraintrain$y, levels = levels(as.factor(as.numeric(curr_pair))))
  curr_stdtrainvalid <- stdtrainvalid[stdtrainvalid$y %in% curr_pair, ]
  curr_stdtrainvalid$y <- as.factor(curr_stdtrainvalid$y) 
  
  # Creating a binary current set for training the model
  curr_stdtraintrain_binary <-  curr_stdtraintrain
  curr_stdtraintrain_binary$y <-  ifelse(curr_stdtraintrain$y == curr_pair[1], 1,0)
  curr_stdtraintrain_binary$y <- factor(curr_stdtraintrain_binary$y)
  
  
  train_control <- trainControl(method="cv", number=10, repeats=3, search="grid")
  grid <- expand.grid(maxdepth=c(1:20))
  model <- train(y~., data=curr_stdtraintrain_binary, method="rpart2", trControl=train_control, tuneGrid=grid)
  
  best_maxdepth[[i]] <- model$bestTune$maxdepth
  
  model_comp <- rpart(y~., data=curr_stdtraintrain_binary, method="class", control=rpart.control(maxdepth=best_maxdepth[[i]]))
  
  rpart.plot(model_comp)
  
  tree_models_comp[[i]] <- prune(model_comp,  cp=model_comp$cp[which.min(model_comp$cp[,"xerror"]),"CP"])
  
  # TEST
  
  curr_stdtrain <- stdtrain[stdtrain$y %in% curr_pair, ]
  curr_stdtrain$y <- as.factor(curr_stdtrain$y)
  
  # Creating a binary current set for training the model
  curr_stdtrain_binary <-  curr_stdtrain
  curr_stdtrain_binary$y <-  ifelse(curr_stdtrain$y == curr_pair[1], 1,0)
  curr_stdtrain_binary$y <- as.factor(curr_stdtrain_binary$y)
  
  model_test <- rpart(y~., data=curr_stdtrain_binary, method="class", control=rpart.control(maxdepth=best_maxdepth[[i]]))

  tree_models_test[[i]] <- prune(model_test,  cp=model_test$cp[which.min(model_test$cp[,"xerror"]),"CP"])
}

## Predictions

# Use the trained classifiers to predict the class of each test instance

voting_valid_preds <- matrix(0, nrow = nrow(stdtrainvalid), ncol = length(class_pairs))
voting_test_preds <- matrix(0, nrow=nrow(stdtest), ncol = length(class_pairs))

weights = c() # setting up the weight lists

for (i in seq_along(class_pairs)){
  
  curr_pair <- class_pairs[[i]]
  
  # VALID
  
  curr_stdtrainvalid <- stdtrainvalid[stdtrainvalid$y %in% curr_pair, ]
  curr_stdtrainvalid$y <- as.factor(curr_stdtrainvalid$y)
  
  
  # Creating a binary current set for validating the model
  curr_stdtrainvalid_binary <-  curr_stdtrainvalid
  curr_stdtrainvalid_binary$y <-  ifelse(curr_stdtrainvalid$y == curr_pair[1], 1,0)
  curr_stdtrainvalid_binary$y <- as.factor(curr_stdtrainvalid_binary$y)
  
  curr_model_comp <- tree_models_comp[[i]]
  
  # Predictions
  
  temp_curr_preds_comp <- predict(curr_model_comp,newdata=curr_stdtrainvalid_binary)
  curr_preds_comp_int <- as.numeric(temp_curr_preds_comp[,2]>.5)
  
  curr_preds_comp <- as.numeric(ifelse(curr_preds_comp_int == 1, curr_pair[1], curr_pair[2] ))
  
  # Weights
  curr_accuracy <- mean(curr_preds_comp == curr_stdtrainvalid$y)
  weights[i] <- curr_accuracy # adding the accuracy of each model to the weight list
  
  voting_valid_preds[stdtrainvalid$y %in% curr_pair, i] <- unlist(curr_preds_comp)
  
  # TEST
  
  curr_model_test <- tree_models_test[[i]]
  
  # Predictions
  
  temp_curr_preds_test <- predict(curr_model_test,newdata=stdtest)
  curr_preds_test_int <- as.numeric(temp_curr_preds_test[,2]>.5)
  
  curr_preds_test <- as.numeric(ifelse(curr_preds_test_int == 1, curr_pair[1], curr_pair[2] ))
  
  
  voting_test_preds[, i] <-unlist(curr_preds_test)
  
}

normalized_weights <- weights/sum(weights)

## AGGREGATION

# FUNCTION
votingFunction <- function(matrix) {
  result <- apply(matrix, 1, function(row) {
    ## UNSTRINGIFY
    # convert the values to numeric
    values_numeric <- as.numeric(row)
    value_counts <- table(values_numeric)
    # get the count of the most common values
    max_count <- max(value_counts)
    # get a list of the most common values
    most_common <- names(value_counts[value_counts == max_count])
    # return the maximum value from the list
    max_value <- max(as.numeric(most_common))
    return(max_value)
  })
  return(result)
}
final_preds_comp <- votingFunction(voting_valid_preds)
final_preds <- votingFunction(voting_test_preds)

# WEIGHTED PREDICTION

weighted_voting_function <- function(matrix) {
  result <- apply(matrix, 1, function(row) {
    # convert the values to numeric
    values_numeric <- as.numeric(row)
    # exclude 0 values from calculation of class_weights
    non_zero_values <- values_numeric[values_numeric != 0]
    if (length(non_zero_values) == 0) {
      # if all values in the row are 0, return 0
      predicted_class <- 0
    } else {
      # calculate class weights for non-zero values
      class_weights <- tapply(normalized_weights[values_numeric != 0], non_zero_values, sum)
      # find the predicted class with the highest sum of weights
      predicted_class <- names(class_weights)[which.max(class_weights)]
    }
    return(as.numeric(predicted_class))
  })
  return(result)
}
final_preds_comp_weighted <- weighted_voting_function(voting_valid_preds)
final_preds_weighted <- weighted_voting_function(voting_test_preds)

### PREDICTION ###
### EVALUATION ###

# Evaluate the accuracy of the final predictions
true_labels <- stdtrainvalid$y
accuracy <- mean(final_preds_comp == true_labels)
print(paste0("Accuracy voting: ", round(accuracy, 3)))

# accuracy with weighted
accuracy_weighted <- mean(final_preds_comp_weighted == true_labels)
print(paste0("Accuracy weighted: ", round(accuracy_weighted, 3)))

### WRITE FILES

pred_model_1v1_tree_w_df <- data.frame(id = 1:length(final_preds_weighted), y = unlist(final_preds_weighted))
write.csv(pred_model_1v1_tree_w_df,file= "predictions/pred_tree_1v1_w.csv", row.names = FALSE)

