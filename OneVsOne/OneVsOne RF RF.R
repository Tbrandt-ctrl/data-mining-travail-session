# Load the libraries
library(ranger)
library(randomForest)
library(summarytools)
library(caret)
library(irr)
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
# Numeric Predictions
create_numeric_predictions <- function (predictions){
  get_col_name <- function(x){ # Create a data frame for the prediction
    return (names(x)[which.max(x)])
  }
  y <- apply(predictions, 1, get_col_name) # Creating the prediction
  return(y)
}
create_numeric_predictions_weighted <- function (predictions, weights){
  get_col_name <- function(x){ # Create a data frame for the prediction
    return (names(x)[which.max(x)])
  }
  y <- apply(predictions * weights, 1, get_col_name) # Weighting the predictions
  return(y)
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
train_index = createDataPartition(train$y, p = .75, list = F) # récupération aléatoire d'index
traintrain <- train[train_index,] # 4000 observations
trainvalid <- train[-train_index,] # 200 observations
# Standardisation
outstd_comp=stdf(traintrain[,!(names(traintrain) %in% namescat)], # standardisation des sets de comparaison
                 trainvalid[,!(names(trainvalid) %in% namescat)])
outstd = stdf(train[,!(names(train) %in% namescat)], test[,!(names(test) %in% namescat)])
stdtraintrain = cbind(traintrain[, (names(traintrain) %in% namescat)], outstd_comp[1])
stdtrainvalid = cbind(trainvalid[, (names(trainvalid) %in% namescat)], outstd_comp[2])
stdtrain =cbind(train[,(names(train) %in% namescat)], outstd[1])
stdtest =cbind(test[,(names(test) %in% namescat)], outstd[2])
### MODEL TRAINING ###
##  Train multiple binary classifiers using random forest algorithm
stdtraintrain$y <- as.factor(stdtraintrain$y)
class_pairs <- combn(levels(stdtraintrain$y), 2, simplify = FALSE)
rf_models_comp <- list()
rf_models_comp_rf <- list()
rf_models <- list()
rf_models_rf <- list()
best_mtrys <-list(5,8,15,10,5,12) # optimisé avec gridsearch avec .75 de train et 200 arbres
best_nodesizes <- list()

for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
  # VALID
  curr_stdtraintrain <- stdtraintrain[stdtraintrain$y %in% curr_pair, ]
  curr_stdtraintrain$y <- factor(curr_stdtraintrain$y, levels = levels(as.factor(as.numeric(curr_pair))))
  # Creating a binary current set for training the model
  curr_stdtraintrain_binary <-  curr_stdtraintrain
  curr_stdtraintrain_binary$y <-  ifelse(curr_stdtraintrain$y == curr_pair[1], 1,0)
  curr_stdtraintrain_binary$y <- as.factor(curr_stdtraintrain_binary$y)
  
  # TUNING
  #  cat("TUNING THE MODEL",curr_pair)
  #   control <-trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
  #   metric <- "Accuracy"
  #  tunegrid <- expand.grid(.mtry=c(1:16), .nodesize = c(5, 10, 20))
  #  rf_gridsearch <- train(y~., data=curr_stdtraintrain_binary, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=200)
  # plot(rf_gridsearch)
  # best_params_mtry[i] <- rf_gridsearch$bestTune$mtry
  # best_params_nodes[i] <- rf_gridsearch$bestTune$nodesize
  
  curr_lowest_err <- 1
  curr_best_node_size <- 5
  
  cat("currently working on this pair: ", curr_pair)
  
  for(j in 5:15){
    model <- rf_models_comp_rf[[i]] <- randomForest(y~., data=curr_stdtraintrain_binary, method="rf", mtry=best_mtrys[[i]], ntree=200, nodesize=j)
    if(mean(model$err.rate[, 1]) < curr_lowest_err){
      cat("working on nodesize: ", j)
      cat("The current best nodesize is: ", curr_best_node_size)
      curr_lowest_err <- mean(model$err.rate[, 1])
      curr_best_node_size <- j
    }
  }
  best_nodesizes[i] <- curr_best_node_size
  rf_models_comp[[i]] <- ranger(y ~., data=curr_stdtraintrain_binary, num.trees=200, importance="permutation",mtry=best_mtrys[[i]],  probability = TRUE, respect.unordered.factors = "partition")
  rf_models_comp_rf[[i]] <- randomForest(y~., data=curr_stdtraintrain_binary, method="rf", mtry=best_mtrys[[i]], ntree=200, nodesize=best_nodesizes[i])
  
  # TEST
  curr_stdtrain <- stdtrain[stdtrain$y %in% curr_pair, ]
  curr_stdtrain$y <- as.factor(curr_stdtrain$y)
  
  # Creating a binary current set for training the model
  curr_stdtrain_binary <-  curr_stdtrain
  curr_stdtrain_binary$y <-  ifelse(curr_stdtrain$y == curr_pair[1], 1,0)
  curr_stdtrain_binary$y <- as.factor(curr_stdtrain_binary$y)
  
  rf_models[[i]] <- ranger(y~., data=curr_stdtrain_binary, num.trees=200, importance="permutation", mtry=best_mtrys[[i]], probability=TRUE,  respect.unordered.factors = "partition")
  rf_models_rf[[i]] <- randomForest(y~., data=curr_stdtrain_binary, method="rf", mtry=best_mtrys[[i]], ntree=200, nodesize=best_nodesizes[i])
}
## Use the trained classifiers to predict the class of each test instance
voting_valid_preds <- matrix(0, nrow = nrow(stdtrainvalid), ncol = length(class_pairs))
voting_test_preds <- matrix(0, nrow=nrow(stdtest), ncol = length(class_pairs))
voting_valid_preds_rf <- matrix(0, nrow = nrow(stdtrainvalid), ncol = length(class_pairs))
voting_test_preds_rf <- matrix(0, nrow=nrow(stdtest), ncol = length(class_pairs))
weights = c() # setting up the weight lists
weights_rf = c()

for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
  
  # VALID
  curr_stdtrainvalid <- stdtrainvalid[stdtrainvalid$y %in% curr_pair, ]
  curr_stdtrainvalid$y <- as.factor(curr_stdtrainvalid$y)
  
  # Creating a binary current set for validating the model
  curr_stdtrainvalid_binary <-  curr_stdtrainvalid
  curr_stdtrainvalid_binary$y <-  ifelse(curr_stdtrainvalid$y == curr_pair[1], 1,0)
  curr_stdtrainvalid_binary$y <- as.factor(curr_stdtrainvalid_binary$y)
  
  curr_model_comp <- rf_models_comp[[i]]
  curr_model_comp_rf <- rf_models_comp_rf[[i]]
  
  curr_preds_comp <- ifelse(create_numeric_predictions(predict(curr_model_comp, data=curr_stdtrainvalid_binary)$predictions) == 1, curr_pair[1], curr_pair[2] )
  
  curr_preds_comp_rf <- ifelse(predict(curr_model_comp_rf, newdata=curr_stdtrainvalid_binary) == 1, curr_pair[1], curr_pair[2])
 
   curr_accuracy <- mean(curr_preds_comp == curr_stdtrainvalid$y)
  weights[i] <- curr_accuracy # adding the accuracy of each model to the weight list
  curr_accuracy_rf <-mean(curr_preds_comp_rf == curr_stdtrainvalid$y)
  weights_rf[i] <- curr_accuracy_rf
  
  voting_valid_preds[stdtrainvalid$y %in% curr_pair, i] <- unlist(curr_preds_comp)
  voting_valid_preds_rf[stdtrainvalid$y %in% curr_pair, i] <- unlist(curr_preds_comp_rf)
  
  # TEST
  curr_model <- rf_models[[i]]
  curr_model_rf <- rf_models_rf[[i]]
  curr_preds <- ifelse(create_numeric_predictions(predict(curr_model, data=stdtest)$predictions)== 1, curr_pair[1], curr_pair[2])
  curr_preds_rf <- ifelse(predict(curr_model_rf, newdata=stdtest)== 1, curr_pair[1], curr_pair[2])
  voting_test_preds[, i] <-unlist(curr_preds)
  voting_test_preds_rf[, i] <-unlist(curr_preds_rf)
}
# Normalize the weights
normalized_weights <- weights/sum(weights)
normalized_weights_rf <- weights_rf/sum(weights_rf  )

# Aggregate the predictions using the voting strategy

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
final_preds_comp_rf <- votingFunction(voting_valid_preds_rf)
final_preds_rf <- votingFunction(voting_test_preds_rf)
# WEIGHTED PREDICTION
# https://ieeexplore.ieee.org/document/9445948
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
final_preds_comp_weighted_rf <- weighted_voting_function(voting_valid_preds_rf)
final_preds_weighted_rf <- weighted_voting_function(voting_test_preds_rf)

### PREDICTION ###
### EVALUATION ###

# Evaluate the accuracy of the final predictions
true_labels <- stdtrainvalid$y
accuracy <- mean(final_preds_comp == true_labels)
print(paste0("Accuracy ranger: ", round(accuracy, 3)))

# accuracy with weighted
accuracy_weighted <- mean(final_preds_comp_weighted == true_labels)
print(paste0("Accuracy ranger weighted: ", round(accuracy_weighted, 3)))
true_labels <- stdtrainvalid$y
accuracy <- mean(final_preds_comp_rf == true_labels)
print(paste0("Accuracy RF: ", round(accuracy, 3)))

# accuracy with weighted
accuracy_weighted <- mean(final_preds_comp_weighted_rf == true_labels)
print(paste0("Accuracy RF weighted: ", round(accuracy_weighted, 3)))

# WRITE FILE
pred_model_1v1_df <- data.frame(id = 1:length(final_preds), y = unlist(final_preds))
write.csv(pred_model_1v1_df,file= "predictions/pred_rf1v1_1.csv", row.names = FALSE)
pred_model_1v1_df_w <- data.frame(id = 1:length(final_preds_weighted), y = unlist(final_preds_weighted))
write.csv(pred_model_1v1_df_w,file= "predictions/pred_rf1v1_1_w.csv", row.names = FALSE)
pred_model_1v1_df_rf <- data.frame(id = 1:length(final_preds_rf), y = unlist(final_preds_rf))
write.csv(pred_model_1v1_df_rf,file= "predictions/pred_rf1v1_1_rf.csv", row.names = FALSE)
pred_model_1v1_df_w_rf <- data.frame(id = 1:length(final_preds_weighted_rf), y = unlist(final_preds_weighted_rf))
write.csv(pred_model_1v1_df_w_rf,file= "predictions/pred_rf1v1_1_w_rf.csv", row.names = FALSE)


best_nodesizes
pred_model_1v1_df <- data.frame(id = 1:length(final_preds), y = unlist(final_preds))
write.csv(pred_model_1v1_df,file= "predictions/pred_rf1v1_1.csv", row.names = FALSE)
pred_model_1v1_df_w <- data.frame(id = 1:length(final_preds_weighted), y = unlist(final_preds_weighted))
write.csv(pred_model_1v1_df_w,file= "predictions/pred_rf1v1_1_w.csv", row.names = FALSE)
pred_model_1v1_df_rf <- data.frame(id = 1:length(final_preds_rf), y = unlist(final_preds_rf))
write.csv(pred_model_1v1_df_rf,file= "predictions/pred_rf1v1_1_rf.csv", row.names = FALSE)
pred_model_1v1_df_w_rf <- data.frame(id = 1:length(final_preds_weighted_rf), y = unlist(final_preds_weighted_rf))
write.csv(pred_model_1v1_df_w_rf,file= "predictions/pred_rf1v1_1_w_rf.csv", row.names = FALSE)

best_nodesizes


rf_models_comp_rf[[i]]$mtry
rf_models_comp_rf[[i]]$call
rf_models_comp_rf[[i]]$type
weights_rf
weights