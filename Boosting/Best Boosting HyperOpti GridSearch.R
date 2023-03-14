set.seed(123)

library(lightgbm)
library(caret)


# https://www.datatechnotes.com/2022/05/lightgbm-multi-class-classification.html

library(summarytools)
library(caret)
library(lightgbm)

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
       as.data.frame(scale(dat2, center=mu , scale=std))
  )
}

# MAUC

getMAUC <- function(predictions, actual_y){ # Use 1:0 versions with 0:3 classes
  aucs <- c()
  for (i in 0:3) {
    converted_predictions <- c(ifelse(predictions == i, 1, 0))
    converted_actuals <- c(ifelse(actual_y == i, 1, 0))
    
    roc_object <- roc(converted_actuals, converted_predictions)
    
    auc <- auc(roc_object)
    
    aucs <- rbind(auc, aucs)
  }
  
  # Calculate the MAUC
  mauc <- mean(aucs)
  
  return(list("MAUC", mauc, TRUE))
}

# KAPPA

getKappa <- function(predictions, actual_y){
  
  data <- cbind(predictions, actual_y)
  kappa <- kappa2(data)$value
  
  return(kappa)
}

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


set.seed(123) # for reproducibility

# Noms des variables à traiter comme catégorielles
namescat=c("country","difflevel")

### VERSION POUR LGBM ### 

lgbtrain <- train
lgbtrain$y <- as.numeric(lgbtrain$y) -1 #pour lgbm

train_index = createDataPartition(lgbtrain$y, p = .80, list = F) # récupération aléatoire d'index


lgbtraintrain = lgbtrain[train_index,]
lgbtrainvalid = lgbtrain[-train_index,]

lgbtraintrain_x = subset(lgbtraintrain, select = -c(y))
lgbtraintrain_y = subset(lgbtraintrain, select = c(y))

lgbtrainvalid_x = subset(lgbtrainvalid, select = -c(y))
lgbtrainvalid_y = subset(lgbtrainvalid, select = c(y))

# Standardisation

outstd = stdf(
  lgbtraintrain_x[, !(names(lgbtraintrain_x) %in% namescat)], 
  lgbtrainvalid_x[, !(names(lgbtrainvalid_x) %in% namescat)]
)
lgbtraintrain_x = cbind(lgbtraintrain_x[, (names(lgbtraintrain_x) %in% namescat)], outstd[[1]])
lgbtrainvalid_x = cbind(lgbtrainvalid_x[, (names(lgbtrainvalid_x) %in% namescat)], outstd[[2]])

# continuer les split et créer des matrices

lgbtraintrain_x = as.matrix(lgbtraintrain_x) 
lgbtraintrain_y = as.matrix(lgbtraintrain_y) 

lgbtrainvalid_x = as.matrix(lgbtrainvalid_x)
lgbtrainvalid_y = as.matrix(lgbtrainvalid_y)

# Créationd d'un fichier d'entrainement dans le bon format pour lightgbm

dtrain = lgb.Dataset(lgbtraintrain_x, categorical_feature = namescat, label=lgbtraintrain_y)
dvalid = lgb.Dataset.create.valid(dtrain, data=lgbtrainvalid_x, label=lgbtrainvalid_y)

# Hyperparamètres

valids <- list(test = dvalid)


# GRID

lgb_grid <- list(
  
  objective = "multiclass", 
  metric = 'multi_logloss', 
  num_class = 4
  
  ,learning_rate = 0.127909
  ,num_leaves = 17
  ,min_data = 97
  ,max_depth = 3749,
  
  is_unbalanced = FALSE
)

lgb_model_cv <- lgb.cv(
  params = lgb_grid,
  data = dtrain,
  valids = valids,
  nrounds = 1000, 
  early_stopping_round = 100,
  eval_freq = 5, 
  nfold = 10L, 
  stratified = TRUE
  
)

best_iter <- lgb_model_cv$best_iter

lgb_model <- lgb.train(
  params= lgb_grid,
  data= dtrain,
  valids=valids,
  nrounds = best_iter,
)


pred_comp = predict(lgb_model, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
pred_comp_y = max.col(pred_comp)-1 


confM <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_comp_y))


print(accuracy <- confM$overall[["Accuracy"]])
print(getKappa(pred_comp_y, lgbtrainvalid_y))
print(getMAUC(pred_comp_y,lgbtrainvalid_y ))