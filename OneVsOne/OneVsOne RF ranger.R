# Load the libraries
library(ranger)
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

##  Train multiple binary classifiers using random forest algorithm

stdtraintrain$y <- as.factor(stdtraintrain$y)

class_pairs <- combn(levels(stdtraintrain$y), 2, simplify = FALSE)
rf_models_comp <- list()

rf_models <- list(
)


for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
  
  
  
  mtry = 13
  
  # VALID
 
  curr_stdtraintrain <- stdtraintrain[stdtraintrain$y %in% curr_pair, ]
  curr_stdtraintrain$y <- factor(curr_stdtraintrain$y, levels = levels(as.factor(as.numeric(curr_pair))))
  
  
  rf_models_comp[[i]] <- ranger(y ~., data=curr_stdtraintrain, num.trees=1500, importance="permutation",mtry=mtry,  probability = TRUE, respect.unordered.factors = "partition")
  
  
  # TEST
  
  curr_stdtrain <- stdtrain[stdtrain$y %in% curr_pair, ]
  curr_stdtrain$y <- as.factor(curr_stdtrain$y)
  
  
  rf_models[[i]] <- ranger(y~., data=curr_stdtrain, num.trees=1500, importance="permutation", mtry=mtry, probability=TRUE,  respect.unordered.factors = "partition")
  
}


## Use the trained classifiers to predict the class of each test instance

voting_valid_preds <- matrix(0, nrow = nrow(stdtrainvalid), ncol = length(class_pairs))
voting_test_preds <- matrix(0, nrow=nrow(stdtest), ncol = length(class_pairs))



weights = c() # setting up the weight lists

for (i in seq_along(class_pairs)){
  
  curr_pair <- class_pairs[[i]]
  
  curr_col_comp <- voting_valid_preds[,i]
  
  # VALID
  
  curr_stdtrainvalid <- stdtrainvalid[stdtrainvalid$y %in% curr_pair, ]
  curr_stdtrainvalid$y <- as.factor(curr_stdtrainvalid$y) 
 
  curr_model_comp <- rf_models_comp[[i]]
  
  # curr_preds_comp <- create_numeric_predictions(predict(curr_model_comp, data=curr_stdtrainvalid)$predictions)
  curr_preds_comp <- create_numeric_predictions(predict(curr_model_comp, data=curr_stdtrainvalid)$predictions)
  
  # curr_accuracy <- mean(curr_preds_comp == curr_stdtrainvalid$y)
  curr_accuracy <- mean(curr_preds_comp == curr_stdtrainvalid$y)
  weights[i] <- curr_accuracy # adding the accuracy of each model to the weight list
  
  # voting_valid_preds[stdtrainvalid$y %in% curr_pair, i] <- unlist(curr_preds_comp)
  
  voting_valid_preds[stdtrainvalid$y %in% curr_pair, i] <- unlist(curr_preds_comp)
  
  # TEST
  
  curr_model <- rf_models[[i]]
  
  curr_preds <- create_numeric_predictions(predict(curr_model, data=stdtest)$predictions)
  
  voting_test_preds[, i] <-unlist(curr_preds)
}

# Normalize the weights

normalized_weights <- weights/sum(weights)


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

# WEIGHTED PREDICTION
# https://ieeexplore.ieee.org/document/9445948

weighted_voting_function <- function(matrix) {
  result <- apply(matrix, 1, function(row) {
    
    # convert the values to numeric
    values_numeric <- as.numeric(row)

    # get weight for each class
    class_weights <- tapply(normalized_weights, values_numeric, sum)
    
    # Find the predicted class with the highest sum of weights
    predicted_class <- names(class_weights)[which.max(class_weights)]
    
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
print(paste0("Accuracy: ", round(accuracy, 3)))

# accuracy with weighted

accuracy_weighted <- mean(final_preds_comp_weighted == true_labels)
print(paste0("Accuracy weighted: ", round(accuracy_weighted, 3)))

# WRITE FILE

pred_model_1v1_df <- data.frame(id = 1:length(final_preds), y = unlist(final_preds)) 
write.csv(pred_model_1v1_df,file= "predictions_model_rf1v1_1.csv", row.names = FALSE) 

pred_model_1v1_df_w <- data.frame(id = 1:length(final_preds_weighted), y = unlist(final_preds_weighted)) 
write.csv(pred_model_1v1_df_w,file= "predictions_model_rf1v1_1_w.csv", row.names = FALSE)


# TESTS

fit.rf = ranger(
  y~., data=stdtraintrain, num.trees=1500
)

fit.rf.tune = csrf(
  y~., training_data=stdtraintrain, test_data=stdtrainvalid, params1=list(num.trees = 1000, mtry=13), params2 = list(num.trees=1500, mtry=13)
)


acc <- mean(fit.rf.tune == stdtrainvalid$y)
