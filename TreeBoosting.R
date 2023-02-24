# https://www.datatechnotes.com/2022/05/lightgbm-multi-class-classification.html

library(irr)
library(yardstick)
library(summarytools)
library(caret)
library(lightgbm)
library(pROC)

### FONCTIONS D'AIDE ### 

# Standardisation

stdf=function(dat1,dat2, dat3)
{
  # input:
  # dat1 = data frame à standardiser avec ses propres moyennes et SD -> Entraînement
  # dat2 = data frame à standardiser en utilisant les moyennes et SD du premier -> Validation
  # output = liste avec les 2 data frames standardisés
  mu=apply(dat1,2,mean)
  std=apply(dat1,2,sd)
  list(as.data.frame(scale(dat1, center=mu , scale=std)),
       as.data.frame(scale(dat2, center=mu , scale=std)),
       as.data.frame(scale(dat3, center=mu , scale=std))
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
  
  return(mauc)
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

# Séparation du dataset train en traintrain et trainvalid
set.seed(123) # for reproducibility

train_index = createDataPartition(train$y, p = .80, list = F) # récupération aléatoire d'index

# Noms des variables à traiter comme catégorielles
namescat=c("country","difflevel")

### VERSION POUR LGBM ### 

lgbtrain <- train
lgbtrain$y <- as.numeric(lgbtrain$y) -1 #pour lgbm

lgbtest <- test

# Splitter le dataset

lgbtraintrain = lgbtrain[train_index,]
lgbtrainvalid = lgbtrain[-train_index,]

lgbtraintrain_x = subset(lgbtraintrain, select = -c(y))
lgbtraintrain_y = subset(lgbtraintrain, select = c(y))

lgbtrainvalid_x = subset(lgbtrainvalid, select = -c(y))
lgbtrainvalid_y = subset(lgbtrainvalid, select = c(y))

lgbtest_x = subset(lgbtest, select = -c(id))


# Standardisation des données

outstd = stdf(
  lgbtraintrain_x[, !(names(lgbtraintrain_x) %in% namescat)], 
  lgbtrainvalid_x[, !(names(lgbtrainvalid_x) %in% namescat)],
  lgbtest_x[, !(names(lgbtest_x) %in% namescat)]
  )
lgbtraintrain_x = cbind(lgbtraintrain_x[, (names(lgbtraintrain_x) %in% namescat)], outstd[[1]])
lgbtrainvalid_x = cbind(lgbtrainvalid_x[, (names(lgbtrainvalid_x) %in% namescat)], outstd[[2]])
lgbtest_x = cbind(lgbtest_x[, (names(lgbtest_x) %in% namescat)], outstd[[3]])
lgbtest = cbind(lgbtest_x, subset(lgbtest, select = c(id)))

# continuer les split et créer des matrices

lgbtraintrain_x = as.matrix(lgbtraintrain_x) 
lgbtraintrain_y = as.matrix(lgbtraintrain_y) 

lgbtrainvalid_x = as.matrix(lgbtrainvalid_x)
lgbtrainvalid_y = as.matrix(lgbtrainvalid_y)

lgbtest = as.matrix(lgbtest)
lgbtest_x = as.matrix(lgbtest_x)


# Créationd d'un fichier d'entrainement dans le bon format pour lightgbm

dtrain = lgb.Dataset(lgbtraintrain_x, categorical_feature = namescat, label=lgbtraintrain_y)
dvalid = lgb.Dataset.create.valid(dtrain, data=lgbtrainvalid_x, label=lgbtrainvalid_y)

### TREE GRADIENT BOOSTING ###

## Pour comparer les résultats ## 

# Entrainement des modèles

params <-  list(objective = "multiclass", 
              metric = 'multi_logloss', 
              num_class = 4)

valids = list(test = dvalid)

modely400 <- lgb.train(params,
                       dtrain, 
                       nrounds = 400L,
                       valids,
                       categorical_feature = namescat,
                    )

modely300 <- lgb.train(params,
                       dtrain, 
                       nrounds = 300L,
                       valids,
                       categorical_feature = namescat,
)

modely141 <- lgb.train(params,
                       dtrain, 
                       nrounds = 141L,
                       valids,
                       categorical_feature = namescat,
                     )




params_random_opti <- list(objective = "multiclass", 
                    metric = 'multi_logloss', 
                    num_class = 4
                    
                    ,learning_rate = 0.127909
                    ,num_leaves = 17
                    ,min_data = 97
                    ,max_depth = 3749
                    ,early_stopping_round = 30
                    )

modelyrandom_opti <- lgb.train(
                      params_random_opti,
                      dtrain, 
                      nrounds = 300L,
                      valids,
                      categorical_feature = namescat
)


# https://neptune.ai/blog/lightgbm-parameters-guide

## Prédidctions ## 

# 400L

pred_comp_400L = predict(modely400, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
pred_comp_y_400L = max.col(pred_comp_400L)-1 

pred_400L = predict(modely400, lgbtest_x, reshape=T) # Prédiction pour 55000 données
pred_y_400L = max.col(pred_400L)-1
pred_y_400L = as.numeric(pred_y_400L) + 1 # Pour les remettre dans le bon format pour Kaggle

conf_400L <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_comp_y_400L), mode = "everything", positive="1")
accuracy_400L <- conf_400L$overall[["Accuracy"]]


# 300L

pred_comp_300L = predict(modely300, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
pred_comp_y_300L = max.col(pred_comp_300L)-1 

pred_300L = predict(modely300, lgbtest_x, reshape=T) # Prédiction pour 55000 données
pred_y_300L = max.col(pred_300L)-1
pred_y_300L = as.numeric(pred_y_300L) + 1 # Pour les remettre dans le bon format pour Kaggle

conf_300L <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_comp_y_300L))
accuracy_300L <- conf_300L$overall[["Accuracy"]]



# 141

pred_comp_141L = predict(modely141, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
pred_comp_y_141L = max.col(pred_comp_141L)-1 

pred_141L = predict(modely141, lgbtest_x, reshape=T) # Prédiction pour 55000 données
pred_y_141L = max.col(pred_141L)-1
pred_y_141L = as.numeric(pred_y_141L) + 1 # Pour les remettre dans le bon format pour Kaggle

conf_141L <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_comp_y_141L))
accuracy_141L <- conf_141L$overall[["Accuracy"]]

# random_opti

pred_comp_random_opti = predict(modelyrandom_opti, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
pred_comp_y_random_opti = max.col(pred_comp_random_opti)-1 

pred_random_opti = predict(modelyrandom_opti, lgbtest_x, reshape=T) # Prédiction pour 55000 données
pred_y_random_opti = max.col(pred_random_opti)-1
pred_y_random_opti = as.numeric(pred_y_random_opti) + 1 # Pour les remettre dans le bon format pour Kaggle

conf_random_opti <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_comp_y_random_opti))
accuracy_random_opti <- conf_random_opti$overall[["Accuracy"]]



### COMPARAISON ### 

## TABLE

comparison_table <- data.frame(
  rbind(
  c(accuracy_400L, 
        getKappa(pred_comp_y_400L, lgbtrainvalid_y),
        getMAUC(pred_comp_y_400L,lgbtrainvalid_y )),
  c(accuracy_300L,
        getKappa(pred_comp_y_300L, lgbtrainvalid_y),
        getMAUC(pred_comp_y_300L,lgbtrainvalid_y )),
  c(accuracy_141L,
    getKappa(pred_comp_y_141L, lgbtrainvalid_y),
    getMAUC(pred_comp_y_141L,lgbtrainvalid_y )),
  c(accuracy_random_opti,
        getKappa(pred_comp_y_random_opti, lgbtrainvalid_y),
        getMAUC(pred_comp_y_random_opti,lgbtrainvalid_y ))
  )
)

names(comparison_table) = c("Accuracy", "MAUC", "Kappa")
row.names(comparison_table) = c("400L", "300L", "141L", "Random Opti")

print(comparison_table)


# IL FAUT OPTIMISER LE NOMBRE D'ARBRES ET LA TAILLE DES ARBRES



### FICHIERS ###
pred_model_gtb_1_df = data.frame(id=test$id, y=pred_y_400L)
write.csv(pred_model_gtb_1_df,file= "pred_model_gtb_1.csv", row.names = FALSE)

pred_model_gtb_2_df = data.frame(id=test$id, y=pred_y_300L)
write.csv(pred_model_gtb_2_df,file= "pred_model_gtb_2.csv", row.names = FALSE)

pred_model_gtb_3_df = data.frame(id=test$id, y=pred_y_141L)
write.csv(pred_model_gtb_3_df,file= "pred_model_gtb_3.csv", row.names = FALSE)

pred_model_gtb_4_df = data.frame(id=test$id, y=pred_y_random_opti)
write.csv(pred_model_gtb_4_df,file= "pred_model_gtb_4.csv", row.names = FALSE)


# le paramètre de rétrécissement 𝜖 n’est
# pas vraiment un paramètres à optimiser. Il suffit de le choisir assez
# petit, habituellement 𝜖 = 0, 001, 0, 01 ou 0, 1.
# OPTIMISATION HYPER PARAMETRES BAYESIEN
# X -> hyperparamètres et performance comme y
# STACKING DE MODELES

### OPTIMISATION ### 






