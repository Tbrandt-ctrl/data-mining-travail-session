library(ranger)
library(summarytools)
library(caret)

### PRÉPARATION DES DONNÉES ###

# Load the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Création de facteurs pour les variables difflevel, country et le y
train$difflevel=factor(train$difflevel)
train$country=factor(train$country)

test$difflevel=factor(test$difflevel)
test$country=factor(test$country)

train$y <- as.factor(train$y)

# Summary
summary(train)
print(dfSummary(train, 
                varnumbers   = FALSE, 
                valid.col    = FALSE, 
                graph.magnif = 0.76),
      method = 'render')

# Séparation du dataset train en traintrain et trainvalid
set.seed(123) # for reproducibility

train_index <- sample(1:nrow(train), round(.8*nrow(train))) # récupération aléatoire d'index

traintrain <- train[train_index,] # 4000 observations
trainvalid <- train[-train_index,] # 1000 observations


### FONCTIONS D'AIDE ### 

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

# Create a dataframe for the RF predictions with numeric Y

create_numeric_predictions <- function (predictions){
  get_col_index <- function(x){ # Create a data frame for the prediction
    return (which.max(x))
  }
  
  y <- apply(predictions, 1, get_col_index) # Creating the prediction
  return(y)
}

create_dataframe <- function (predictions, test_data){
  
  y <- create_numeric_predictions(predictions)
  
  # dataset <- cbind(predictions, y) # Create a dataset with everything to check it is right
  dataframe <- data.frame(id=test_data$id, y=y) # Create a dataframe to be exported to csv
  
  return (dataframe)
}


### MODÈLES ###

##  Random Forests ##

# Basic RF with Mtry=13

model_rf_1_comp <- ranger(y ~., data=traintrain, num.trees=1500, importance = "impurity",mtry=13,  probability = TRUE, respect.unordered.factors = "partition")
model_rf_1 <- ranger(y ~., data=train, num.trees=1500, importance = "impurity",mtry=13,  probability = TRUE, respect.unordered.factors = "partition")


pred_model_rf_1_comp <- predict(model_rf_1_comp, data=trainvalid)$predictions # 1000 prédictions
pred_model_rf_1_test <- predict(model_rf_1, data=test)$predictions # 55000 prédictions


num_pred_model_rf_1_comp <- create_numeric_predictions(pred_model_rf_1_comp)
num_pred_model_rf_1_test <- create_numeric_predictions(pred_model_rf_1_test)

## Standardisation et RF

# Standardiser

namesfactor=names(train)[sapply(train, class)=="factor"] # préparation générale


outstd_comp=stdf(traintrain[,!(names(traintrain) %in% namesfactor)], # standardisation des sets de comparaison
            trainvalid[,!(names(trainvalid) %in% namesfactor)])

stdtrain_sans_facteurs = scale(train[,!(names(train) %in% namesfactor)],# standardisation du training set
                               center=apply(train[,!(names(train) %in% namesfactor)],2,mean) , 
                               scale=apply(train[,!(names(train) %in% namesfactor)],2,sd))


cols_to_remove <- c(names(train)[sapply(train, is.factor)], "id") # prépration de la standardisation du test set

test_sans_facteurs_sans_id <- test[, !(names(test) %in% cols_to_remove)]
test_avec_facteurs_et_id = test[,(names(test) %in% cols_to_remove)]


stdtest_sans_facteurs_sans_id = scale(test_sans_facteurs_sans_id, 
                                      center=apply(train[,!(names(train) %in% namesfactor)],2,mean) , 
                                      scale=apply(train[,!(names(train) %in% namesfactor)],2,sd))
  
stdtest =  cbind(stdtest_sans_facteurs_sans_id,test_avec_facteurs_et_id) # liaison test standardisé

stdtrain = cbind(stdtrain_sans_facteurs, train[,(names(train) %in% namesfactor)]) # liaison train standardisé

stdtraintrain=cbind(traintrain[,(names(traintrain) %in% namesfactor)],outstd_comp[[1]]) # liaison train standardisé de comparaison
stdtrainvalid=cbind(trainvalid[,(names(trainvalid) %in% namesfactor)],outstd_comp[[2]]) # liaison valid standardisé de comparaison

# Forêt avec set standardisé

model_rf_2_comp <- ranger(y ~., data=stdtraintrain, num.trees=1500, importance = "impurity",mtry=13,  probability = TRUE, respect.unordered.factors = "partition")
model_rf_2 <- ranger(y ~., data=stdtrain, num.trees=1500, importance = "impurity",mtry=13,  probability = TRUE, respect.unordered.factors = "partition")
model_rf_2 # OOB  0.2217346 

pred_model_rf_2_comp <- predict(model_rf_2_comp, data=stdtrainvalid)$predictions # 1000 prédictions
pred_model_rf_2_test <- predict(model_rf_2, data=stdtest)$predictions # 55000 prédictions

num_pred_model_rf_2_comp <- create_numeric_predictions(pred_model_rf_2_comp)
num_pred_model_rf_2_test <- create_numeric_predictions(pred_model_rf_2_test)

## Boosting d'arbre ## 

## Boosting de forêt ##

## One-Vs-One ##

factor_levels <- levels(traintrain$y)
for(level in factor_levels){
  print(level)
}

### COMPARAISON DE MODÈLES ### 

## Préparation

# create a function to evaluate classification models
evaluate_model <- function(predictions, reference, num_classes, model_name) {

  pred_factor <- as.factor(predictions)
  
  # Create confusion matrix
  conf_matrix <- confusionMatrix(data = pred_factor, reference = reference)
  
  # Extract accuracy
  accuracy <- conf_matrix$overall["Accuracy"]
  
  #calculate Cohen's Kappa
  library(irr)
  data <- cbind(num_pred_model_rf_1_comp, trainvalid$y)
  kappa <- kappa2(data)$value
  
  # create dataframe with evaluation metrics
  eval_df <- data.frame(
    Accuracy = I(accuracy),
    Kappa = I(kappa)
  )
  
  
  # Set row names for each class
  row.names(eval_df) <- c(model_name)
  
  return(eval_df) # On se limite à ça pour le moment car le reste ne semble fonctionner
}

## Comparaison

evaluate_model(num_pred_model_rf_1_comp, trainvalid$y, 4, "RF_1")
evaluate_model(num_pred_model_rf_2_comp, stdtrainvalid$y, 4, "RF_2_stand")




### FICHIERS ###
pred_model_rf_1_df <- create_dataframe(pred_model_rf_1_test, test) 
write.csv(pred_model_rf_1_df,file= "predictions_model_rf1.csv", row.names = FALSE)

pred_model_rf_2_df <- create_dataframe(pred_model_rf_2_test, test) 
write.csv(pred_model_rf_2_df,file= "predictions_model_rf2.csv", row.names = FALSE) 



