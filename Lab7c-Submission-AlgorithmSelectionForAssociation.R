
.libPaths()
lapply(.libPaths(), list.files)

if (require("arules")) {
  require("arules")
} else {
  install.packages("arules", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## arulesViz ----
if (require("arulesViz")) {
  require("arulesViz")
} else {
  install.packages("arulesViz", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## tidyverse ----
if (require("tidyverse")) {
  require("tidyverse")
} else {
  install.packages("tidyverse", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readxl ----
if (require("readxl")) {
  require("readxl")
} else {
  install.packages("readxl", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## knitr ----
if (require("knitr")) {
  require("knitr")
} else {
  install.packages("knitr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## lubridate ----
if (require("lubridate")) {
  require("lubridate")
} else {
  install.packages("lubridate", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plyr ----
if (require("plyr")) {
  require("plyr")
} else {
  install.packages("plyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## RColorBrewer ----
if (require("RColorBrewer")) {
  require("RColorBrewer")
} else {
  install.packages("RColorBrewer", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
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
train <- read_csv("data/train.csv")
View(train)
# Load the "train.csv" dataset
data <- read_csv("data/train.csv")

# Data Preprocessing

# Handling Missing Values
data <- na.omit(data)  # Remove rows with missing values

# Encoding Categorical Variables
data$Gender <- as.factor(data$Gender)
data$Ever_Married <- as.factor(data$Ever_Married)
data$Graduated <- as.factor(data$Graduated)
data$Profession <- as.factor(data$Profession)
data$Spending_Score <- as.factor(data$Spending_Score)
data$Var_1 <- as.factor(data$Var_1)
data$Segmentation <- as.factor(data$Segmentation)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
splitIndex <- createDataPartition(data$Segmentation, p = 0.7, list = FALSE)
trainData <- data[splitIndex, ]
testData <- data[-splitIndex, ]

# Train the Decision Tree Model
segmentation_model_rpart <- rpart(Segmentation ~ Gender + Ever_Married + Age + Graduated + Profession + Work_Experience + Spending_Score + Family_Size + Var_1,
                                  data = trainData)

# Display the model's details
print(segmentation_model_rpart)

# Make predictions
predictions <- predict(segmentation_model_rpart,
                       testData,
                       type = "class")

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, testData$Segmentation)
print(confusion_matrix)

# Visualize the confusion matrix
fourfoldplot(as.table(confusion_matrix$confusion), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

