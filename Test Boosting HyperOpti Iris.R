
set.seed(100)

library(lightgbm)
library(caret)

# We load the default iris dataset shipped with R
data(iris)

# We must convert factors to numeric
# They must be starting from number 0 to use multiclass
# For instance: 0, 1, 2, 3, 4, 5...
iris$Species <- as.numeric(as.factor(iris$Species)) - 1L

# We cut the data set into 80% train and 20% validation
# The 10 last samples of each class are for validation



# Method 1 of training
#params <- list(
#  objective = "multiclass",
#  metric = 'multi_logloss', 
#   num_class = 3L
  
 
# )
# model <- lgb.train(
#   params
#  , dtrain
# , 100L
#  , valids
# )

# Method 2 random grid search

iterations <- 120L

best_accuracy <- 0.6
best_hyperparams <- list()

for (i in 100:iterations){
    print(paste0("iteration count ",i))
    tryCatch({
      
      # Preparing the data
      
      train_index = createDataPartition(iris$Species, p = .80, list = F) # récupération aléatoire d'index
      
      
      train <- as.matrix(iris[train_index,])
      valid <- as.matrix(iris[-train_index,])
      
      train_x <-subset(train, select = -c(Species))
      train_y <-subset(train, select = c(Species))
      
      valid_x <-subset(valid, select = -c(Species))
      valid_y <-subset(valid, select = c(Species))
      
      
      dtrain <- lgb.Dataset(data = train_x, label = train_y)
      dvalid <- lgb.Dataset.create.valid(dtrain, data = valid_x, label = valid_y)
      
      valids <- list(test = dvalid)
      
      hyperparams <- list( 
                        objective = "multiclass",
                        metric = 'multi_logloss', 
                        num_class = 3L
                        )
      
      # Random Hyperparameters
      
      hyperparams$learning_rate <- runif(1, min=0, max=1)
      hyperparams$num_leaves <- sample(20:300, size=1)
      hyperparams$min_data <- sample(10:100, size=1)
      hyperparams$max_depth <- sample(10:10000, size=1)
      print(hyperparams, iterations)
      
      # Random Model
      
      random_model <- lgb.train( hyperparams
                                 , dtrain
                                 , iterations
                                 , valids
                                 )
      
      # Random predict
      
      pred_random = predict(random_model, valid_x, reshape=T) # Prediction pour 1000 données
      pred_random_y = max.col(pred_random)-1 
      
      
      # Accuracy
      
      confM <- confusionMatrix(as.factor(valid_y), as.factor(pred_random_y))
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





