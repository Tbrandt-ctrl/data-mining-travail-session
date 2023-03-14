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


### ITERATIONS

iterations <- 10000

best_accuracy <- 0.78
best_hyperparams <- listM()

for (i in 1:iterations){
  print(paste0("iteration count ",i))
  tryCatch({
    
    # Séparation du dataset train en traintrain et trainvalid
    
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
    
    hyperparams <- list( 
      objective = "multiclass",
      metric = 'multi_logloss', 
      num_class = 4L
    )
    
    # Random Hyperparameters
    
    hyperparams$num_iter <- sample(200:300, size=1)
    hyperparams$learning_rate <- runif(1, min=0, max=1)
    hyperparams$num_leaves <- sample(20:300, size=1)
    hyperparams$min_data <- sample(10:100, size=1)
    hyperparams$max_depth <- sample(10:5000, size=1)
    hyperparams$early_stopping_round <- 30 # 10% of number iter
    print(hyperparams, iterations)
    
    # Random Model
    
    random_model <- lgb.train( hyperparams
                               , dtrain
                               , iterations
                               , valids
                               , categorical_feature = namescat,
                               
    )
    
    # Random predict
    
    pred_random = predict(random_model, lgbtrainvalid_x, reshape=T) # Prediction pour 1000 données
    pred_random_y = max.col(pred_random)-1 
    
    # Accuracy
    
    confM <- confusionMatrix(as.factor(lgbtrainvalid_y), as.factor(pred_random_y), mode = "everything", positive="1")
    accuracy <- confM$overall[["Accuracy"]]
    
    if(accuracy > best_accuracy){
      best_accuracy <- accuracy
      best_hyperparams <- hyperparams
    }
    
  }, 
  error=function(cond){
    
    message(cond)
    
    return(NA)
  },
  finally={
    message("Wow it worked")
  })
}

print(best_accuracy)
print(best_hyperparams)

