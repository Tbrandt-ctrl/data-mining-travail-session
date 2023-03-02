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

train$y <- as.factor(train$y)

# Noms des variables à traiter comme catégorielles
namescat=c("country","difflevel", "y")

# Split the dataset into training and validation

train_index = createDataPartition(train$y, p = .80, list = F) # récupération aléatoire d'index

traintrain <- train[train_index,] # 4000 observations
trainvalid <- train[-train_index,] # 1000 observations

# Standardisation

outstd=stdf(traintrain[,!(names(traintrain) %in% namescat)], # standardisation des sets de comparaison
                 trainvalid[,!(names(trainvalid) %in% namescat)])

stdtraintrain = cbind(traintrain[, (names(traintrain) %in% namescat)], outstd[1])
stdtrainvalid = cbind(trainvalid[, (names(trainvalid) %in% namescat)], outstd[2])

### MODEL TRAINING ###

##  Train multiple binary classifiers using random forest algorithm

class_pairs <- combn(levels(stdtraintrain$y), 2, simplify = FALSE)
rf_models <- list()

for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
 
  curr_stdtraintrain <- stdtraintrain[stdtraintrain$y %in% curr_pair, ]
  curr_stdtraintrain$y <- as.factor(curr_stdtraintrain$y) 
  
  rf_models[[i]] <- ranger(y ~., data=curr_stdtraintrain, num.trees=1500, importance="permutation",mtry=13,  probability = FALSE, respect.unordered.factors = "partition")
  
}

## Use the trained classifiers to predict the class of each test instance

voting_valid_preds <- matrix(0, nrow = nrow(stdtrainvalid), ncol = length(class_pairs))
colnames(voting_valid_preds) <- c("1:2", "1:3","1:4", "2:3" , "2:4", "3:4")

for (i in seq_along(class_pairs)){
  curr_pair <- class_pairs[[i]]
  
  curr_stdtrainvalid <- stdtrainvalid[stdtrainvalid$y %in% curr_pair, ]
  curr_stdtrainvalid$y <- as.factor(curr_stdtrainvalid$y) 
 
  curr_model <- rf_models[[i]]
  
  dummy_id <- curr_stdtrainvalid[,-1:-length(curr_stdtrainvalid)]
  
  curr_preds <- predict(curr_model, data=curr_stdtrainvalid)$predictions
  # PERTE D'INFORMATIONS SUR L'ID AVANT LES PREDICTION
  
  dummy_id$preds <- curr_preds
  
  dummy_id <- as.list(dummy_id)
  num_curr_preds <- dummy_id
  
  length(voting_valid_preds[stdtrainvalid$y %in% curr_pair, i])
  
  voting_valid_preds[stdtrainvalid$y %in% curr_pair, i] <- unlist(num_curr_preds)
}


# Aggregate the predictions using the voting strategy

final_preds <- apply(voting_valid_preds, 1, function(x) {
  if (sum(x == 1) >= sum(x == 2) & sum(x == 1) >= sum(x == 3) & sum(x == 1) >= sum(x == 4)) {
    return("1")
  } else if (sum(x == 2) >= sum(x == 1) & sum(x == 2) >= sum(x == 3) & sum(x == 2) >= sum(x == 4)) {
    return("2")
  } else if (sum(x == 3) >= sum(x == 1) & sum(x == 3) >= sum(x == 2) & sum(x == 3) >= sum(x == 4)) {
    return("3")
  } else {
    return("4")
  }
})



### PREDICTION ### 

### EVALUATION ### 

# Evaluate the accuracy of the final predictions

true_labels <- stdtrainvalid$y
accuracy <- mean(final_preds == true_labels)
print(paste0("Accuracy: ", round(accuracy, 3)))
