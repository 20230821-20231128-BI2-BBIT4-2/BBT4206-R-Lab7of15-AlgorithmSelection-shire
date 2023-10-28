.libPaths()
lapply(.libPaths(), list.files)

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 1. Install and Load the Required Packages ----
## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## corrplot ----
if (require("corrplot")) {
  require("corrplot")
} else {
  install.packages("corrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggcorrplot ----
if (require("ggcorrplot")) {
  require("ggcorrplot")
} else {
  install.packages("ggcorrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load the Dataset ----
library(readr)
train <- read_csv("data/train.csv")
View(train)

train <-
  read_csv("data/train.csv",
           col_types =
             cols(ID = col_double(),
                  Gender = col_character(),
                  Ever_Married = col_character(),
                  Age = col_double(),
                  Graduated = col_character(),
                  Profession = col_character(),
                  Work_Experience = col_double(),
                  Family_Size = col_double(),
                  Spending_Score = col_character(),
                  Var_1 = col_character(),
                  Segmentation = col_character()))


train$Profession <- factor(train$Profession)

str(train)
dim(train)
head(train)
summary(train)

# STEP 3. Check for Missing Data and Address it ----
# Are there missing values in the dataset?
any_na(train)

# How many?
n_miss(train)

# What is the proportion of missing data in the entire dataset?
prop_miss(train)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(train)

# Which variables contain the most missing values?
gg_miss_var(train)

# Which combinations of variables are missing together?
gg_miss_upset(train)

# Where are missing values located (the shaded regions in the plot)?
vis_miss(train) +
  theme(axis.text.x = element_text(angle = 80))

## OPTION 1: Remove the observations with missing values ----
# We can decide to remove all the observations that have missing values
# as follows:
train_removed_obs <- train %>% filter(complete.cases(.))

train_removed_obs <-
  train %>%
  dplyr::filter(complete.cases(.))

# The initial dataset had 21,120 observations and 16 variables
dim(train)

# The filtered dataset has 16,205 observations and 16 variables
dim(train_removed_obs)

# Are there missing values in the dataset?
any_na(train_removed_obs)

## OPTION 2: Remove the variables with missing values ----
# Alternatively, we can decide to remove the 2 variables that have missing data
train_removed_vars <-
  train %>%
  dplyr::select(-Work_Experience, -Family_Size)

# The initial dataset had 21,120 observations and 16 variables
dim(train)

# The filtered dataset has 21,120 observations and 14 variables
dim(train_removed_vars)

# Are there missing values in the dataset?
any_na(train_removed_vars)

## OPTION 3: Perform Data Imputation ----

# CAUTION:
# 1. Avoid Over-imputation:
# Be cautious when imputing dates, especially if it is
# Missing Not at Random (MNAR).
# Over-Imputing can introduce bias into your analysis. For example, if dates
# are missing because of a specific event or condition, imputing dates might
# not accurately represent the data.

# 2. Consider the Business Context:
# Dates often have a significant business or domain context. Imputing dates
# may not always be appropriate, as it might distort the interpretation of
# your data. For example, imputing order dates could lead to incorrect insights
# into seasonality trends.

# library(mice) # nolint
# somewhat_correlated_variables <- quickpred(airbnb_cape_town, mincor = 0.3) # nolint

# airbnb_cape_town_imputed <-
#   mice(airbnb_cape_town, m = 11, method = "pmm",
#        seed = 7, # nolint
#        predictorMatrix = somewhat_correlated_variables)

# The choice left is between OPTION 1 and OPTION 2:
# Considering that the 2 variables had 23.3% missing data each,
# we decide to remove the observations that have the missing data (OPTION 1)
# as opposed to removing the entire variable just because 23.3% of its values
# are missing (OPTION 2).

# STEP 4. Perform EDA and Feature Selection ----
## Compute the correlations between variables ----
# We identify the correlated variables because it is these correlated variables
# that can then be used to identify the clusters.

# Create a correlation matrix
# Option 1: Basic Table
cor(train_removed_obs[, c(1, 4, 7, 9)]) %>%
  View()

# Option 2: Basic Plot
cor(train_removed_obs[, c(1, 4, 7, 9)]) %>%
  corrplot(method = "square")

# Option 3: Fancy Plot using ggplot2
corr_matrix <- cor(train_removed_obs[, c(1, 4, 7, 9)])

p <- ggplot2::ggplot(data = reshape2::melt(corr_matrix),
                     ggplot2::aes(Var1, Var2, fill = value)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = label_wrap(label, width = 10)),
                     size = 4) +
  ggplot2::theme_minimal() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

# The correlation plot shows a -0.06 correlation between the price and the
# reviews_per_month. This is worth investigating further if the intention
# of the business is to create clusters based on price.

# Room_type, neighbourhood, date and other non-numeric variables and
# categorical variables are not included in the correlation, but they can be
# used as an additional dimension when plotting the scatter plot during EDA.

## Plot the scatter plots ----
# A scatter plot to show How long a person has worked against Type of variable
# per room type
ggplot(train_removed_obs,
       aes(Work_Experience, Var_1,
           color = Age,
           shape = Spending_Score)) +
  geom_point(alpha = 0.5) +
  xlab("How long a person has worked") +
  ylab("Type of variable")


# A scatter plot to show Total Spending Amount against Characteristics of Each Segment
# per review year
ggplot(train_removed_obs,
       aes(Spending_Score, Segmentation,
           color = Graduated, shape = Segmentation)) +
  geom_point(alpha = 0.5) +
  xlab("Total Spending Amount") +
  ylab("Characteristics of Each Segment")


## Transform the data ----
# The K Means Clustering algorithm performs better when data transformation has
# been applied. This helps to standardize the data making it easier to compare
# multiple variables.

summary(train_removed_obs)
model_of_the_transform <- preProcess(train_removed_obs, method = c("scale", "center"))
print(model_of_the_transform)
train_removed_obs_std <- predict(model_of_the_transform, train_removed_obs)
summary(train_removed_obs_std)  # Use 'train_removed_obs_std' here, not 'train_obs_std'
sapply(train_removed_obs_std[, c(1, 4, 7, 9)], sd)

## Select the features to use to create the clusters ----
# OPTION 1: Use all the numeric variables to create the clusters
train_vars <- train_removed_obs_std[, c(1, 4, 7, 9)]

train_vars <-
  train_removed_obs_std[, c("Age",
                            "Work_Experience")]

# STEP 5. Create the clusters using the K-Means Clustering Algorithm ----
# We start with a random guess of the number of clusters we need
set.seed(7)
kmeans_cluster <- kmeans(train_vars, centers = 3, nstart = 20)

# We then decide the maximum number of clusters to investigate
n_clusters <- 8

# Initialize total within sum of squares error: wss
wss <- numeric(n_clusters)

set.seed(7)

# Investigate 1 to n possible clusters (where n is the maximum number of 
# clusters that we want to investigate)
for (i in 1:n_clusters) {
  # Use the K Means cluster algorithm to create each cluster
  kmeans_cluster <- kmeans(train_vars, centers = i, nstart = 20)
  # Save the within cluster sum of squares
  wss[i] <- kmeans_cluster$tot.withinss
}

## Plot a scree plot ----
# The scree plot should help you to note when additional clusters do not make
# any significant difference (the plateau).
wss_df <- tibble(clusters = 1:n_clusters, wss = wss)

scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
  geom_point(size = 4) +
  geom_line() +
  scale_x_continuous(breaks = c(2, 4, 6, 8)) +
  xlab("Number of Clusters")



# OPTION 2: Use only the most significant variables to create the clusters
# This can be informed by feature selection, or by the business case.

# Suppose that the business case is that we need to know the clusters that
# are related to the number of listings a host owns against the listings'
# popularity (measured by number of reviews).

# We need to find the ideal number of listings to own without negatively
# impacting the popularity of the listing.
scree_plot

# We can add guides to make it easier to identify the plateau (or "elbow").
scree_plot +
  geom_hline(
    yintercept = wss,
    linetype = "dashed",
    col = c(rep("#000000", 5), "#FF0000", rep("#000000", 2))
  )

# The plateau is reached at 6 clusters.
# We therefore create the final cluster with 6 clusters
# (not the initial 3 used at the beginning of this STEP.)
k <- 6
set.seed(7)
# Build model with k clusters: kmeans_cluster
kmeans_cluster <- kmeans(train_vars, centers = k, nstart = 20)

# STEP 6. Add the cluster number as a label for each observation ----
train_removed_obs$cluster_id <- factor(kmeans_cluster$cluster)

## View the results by plotting scatter plots with the labelled cluster ----
ggplot(train_removed_obs, aes(Work_Experience, Var_1,
                              color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("How long a person has worked") +
  ylab("Type of variable")

ggplot(train_removed_obs,
       aes(Spending_Score, Segmentation, color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("Total Spending Amount") +
  ylab("Characteristic of Each Segment")