.libPaths()
lapply(.libPaths(), list.files)


if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 1. Install and Load the Required Packages ----
## stats ----
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS ----
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## glmnet ----
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab ----
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## rpart ----
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


library(readr)
mtcars <- read_csv("data/mtcars.csv")
View(mtcars)

# A. Linear Algorithms ----
## 1. Linear Regression ----
### 1.a. Linear Regression using Ordinary Least Squares without caret ----

#### Load and split the dataset ----
data(mtcars)

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(mtcars$`mpg`,
                                   p = 0.8,
                                   list = FALSE)
mtcars_train <- mtcars[train_index, ]
mtcars_test <- mtcars[-train_index, ]

#### Train the model ----
mtcars_model_lm <- lm(disp ~ ., mtcars_train)

#### Display the model's details ----
print(mtcars_model_lm)

#### Make predictions ----
predictions <- predict(mtcars_model_lm, newdata = mtcars_test)


#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((mtcars_test$mpg - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
ssr <- sum((mtcars_test$mpg - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
sst <- sum((mtcars_test$mpg - mean(mtcars_test$mpg))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----

r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----

absolute_errors <- abs(predictions - mtcars_test$mpg)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

### 1.b. Linear Regression using Ordinary Least Squares with caret ----
#### Load and split the dataset ----
data(mtcars)

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(mtcars$mpg,
                                   p = 0.8,
                                   list = FALSE)
mtcars_train <- mtcars[train_index, ]
mtcars_test <- mtcars[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
mtcars_caret_model_lm <- train(mpg ~ ., data = mtcars_train,
                                       method = "lm", metric = "RMSE",
                                       preProcess = c("center", "scale"),
                                       trControl = train_control)

#### Display the model's details ----
print(mtcars_caret_model_lm)

#### Make predictions ----
predictions <- predict(boston_housing_caret_model_lm,
                       boston_housing_test[, 1:13])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((boston_housing_test$medv - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----

ssr <- sum((boston_housing_test$medv - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----

sst <- sum((boston_housing_test$medv - mean(boston_housing_test$medv))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----

absolute_errors <- abs(predictions - boston_housing_test$medv)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))
## 2. Logistic Regression ----
### 2.a. Logistic Regression without caret ----
#### Load and split the dataset ----
data(mtcars)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(mtcars$mpg,
                                   p = 0.7,
                                   list = FALSE)
mtcars_train <- mtcars[train_index, ]
mtcars_test <- mtcars[-train_index, ]

#### Train the model ----
vs_model_glm <- glm(vs ~ ., data = mtcars_train,
                          family = binomial(link = "logit"))

#### Display the model's details ----
print(vs_model_glm)

#### Make predictions ----
probabilities <- predict(vs_model_glm, mtcars_test,
                         type = "response")
print(probabilities)
predictions <- ifelse(probabilities > 0.5, "0", "1")
print(predictions)

#### Display the model's evaluation metrics ----
table(predictions, mtcars_test$mpg)

### 2.b. Logistic Regression with caret ----
#### Load and split the dataset ----
data(mtcars)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(mtcars$mpg,
                                   p = 0.7,
                                   list = FALSE)
mtcars_train <- mtcars[train_index, ]
mtcars_test <- mtcars[-train_index, ]

#### Train the model ----

train_control <- trainControl(method = "cv", number = 5)

set.seed(7)
mpg_caret_model_logistic <-
  train(mpg ~ ., data = mtcars_train,
        method = "regLogistic", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(mpg_caret_model_logistic)

#### Make predictions ----
predictions <- predict(mpg_caret_model_logistic,
                       mtcars_test[, 1:8])

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         mtcars_test[, 1:9]$diabetes)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")
#3
library(readr)
HeartDiseaseTrain_Test <- read_csv("data/HeartDiseaseTrain-Test.csv")
View(HeartDiseaseTrain_Test)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(HeartDiseaseTrain_Test$thalassemia,
                                   p = 0.7,
                                   list = FALSE)
HeartDiseaseTrain_Test_train <- HeartDiseaseTrain_Test[train_index, ]
HeartDiseaseTrain_Test_test <- HeartDiseaseTrain_Test[-train_index, ]

#### Train the model ----
thalassemia_model_lda <- lda(thalassemia ~ ., data = HeartDiseaseTrain_Test_train)

#### Display the model's details ----
print(thalassemia_model_lda)

#### Make predictions ----

predictions <- predict(thalassemia_model_lda, HeartDiseaseTrain_Test_test, type = "response")$class

#### Display the model's evaluation metrics ----
table(predictions, HeartDiseaseTrain_Test_test$thalassemia)


### 3.b.  Linear Discriminant Analysis with caret ----
#### Load and split the dataset ----


# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(HeartDiseaseTrain_Test$thalassemia,
                                   p = 0.7,
                                   list = FALSE)
HeartDiseaseTrain_Test_train <- HeartDiseaseTrain_Test[train_index, ]
HeartDiseaseTrain_Test_test <- PimaIndiansDiabetes[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "LOOCV")
thalassemia_caret_model_lda <- train(thalassemia ~ .,
                                  data = HeartDiseaseTrain_Test_train,
                                  method = "lda", metric = "Accuracy",
                                  preProcess = c("center", "scale"),
                                  trControl = train_control)

#### Display the model's details ----
print(thalassemia_caret_model_lda)

#### Make predictions ----
predictions <- predict(thalassemia_caret_model_lda,
                       HeartDiseaseTrain_Test_test)
# Include all predictor variables in the test data
predictions <- predict(thalassemia_caret_model_lda, newdata = HeartDiseaseTrain_Test_test)


#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         HeartDiseaseTrain_Test_test[, 1:9]$thalassememia)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 4. Regularized Linear Regression ----
### 4.a. Regularized Linear Regression Classification Problem without CARET ----
data(HeartDiseaseTrain_Test)
x <- as.matrix(HeartDiseaseTrain_Test[, 1:8])
y <- as.matrix(HeartDiseaseTrain_Test[, 9])

#### Train the model ----
thalassememia_model_glm <- glmnet(x, y, family = "binomial",
                             alpha = 0.5, lambda = 0.001)

#### Display the model's details ----
print(thalassememia_model_glm)

#### Make predictions ----
predictions <- predict(thalassamemia_model_glm, x, type = "class")

#### Display the model's evaluation metrics ----
table(predictions, HeartDiseaseTrain_Test$thalassemia)

### 4.b. Regularized Linear Regression Regression Problem without CARET ----
#### Load the dataset ----
data(HeartDiseaseTrain_Test)
HeartDiseaseTrain_Test$target <- # nolint: object_name_linter.
  as.numeric(as.character(HeartDiseaseTrain_Test$target))
x <- as.matrix(HeartDiseaseTrain_Test[, 1:13])
y <- as.matrix(HeartDiseaseTrain_Test[, 14])

#### Train the model ----
HeartDiseaseTrain_Test_model_glm <- glmnet(x, y, family = "gaussian",
                                   alpha = 0.5, lambda = 0.001)

#### Display the model's details ----
print(HeartDiseaseTrain_Test_model_glm)

#### Make predictions ----
predictions <- predict(HeartDiseaseTrain_Test_model_glm, x, type = "link")

#### Display the model's evaluation metrics ----
mse <- mean((y - predictions)^2)
print(mse)
##### RMSE ----
rmse <- sqrt(mean((y - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
ssr <- sum((y - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
sst <- sum((y - mean(y))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
absolute_errors <- abs(predictions - y)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))


