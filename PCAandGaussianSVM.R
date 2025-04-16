library(png)
library(jpeg)
library(dplyr)
library(tidyr)
library(caret)
library(ggcorrplot)
library(ggplot2)
library(gridExtra)
library(kableExtra)
library(knitr)
library(purrr)
library(kernlab)
library(BiocManager)
library(readr)
library(doParallel)
if (!requireNamespace("EBImage", quietly = TRUE)) {
  BiocManager::install("EBImage")
}
library(EBImage)
set.seed(100)

trainNoFeatureEngineering <- read_csv("Stat380Final/Prepped_Data/trainNoFeatureEngineering.csv")
testNoFeatureEngineering <- read_csv("Stat380Final/Prepped_Data/testNoFeatureEngineering.csv")


# It’s common to remove non-numeric columns when performing PCA.
# Here we automatically select only numeric columns.
numeric_data <- trainNoFeatureEngineering %>% 
  select(where(is.numeric))

# Perform PCA, scaling the variables since they may be on different scales
pca_result <- prcomp(numeric_data, scale. = TRUE)

# Calculate the proportion of variance explained for each principal component.
# pca_result$sdev holds the standard deviations for each PC.
variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cumulative_variance <- cumsum(variance_explained)

# Create a data frame for plotting
pca_df <- data.frame(
  PC = 1:length(variance_explained),
  VarianceExplained = variance_explained,
  CumulativeVariance = cumulative_variance
)

# Optionally, determine the number of components needed to explain at least 90% of the variance.
target_threshold <- 0.95
n_components <- which(cumulative_variance >= target_threshold)[1]
message(sprintf("Number of components to reach at least %.0f%% variance explained: %d", 
                target_threshold * 100, n_components))

# Create a scree plot with both the bar chart (variance explained per PC) and a line for cumulative variance.
scree_plot <- ggplot(pca_df, aes(x = PC)) +
  geom_bar(aes(y = VarianceExplained), stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_line(aes(y = CumulativeVariance), color = "red", size = 1) +
  geom_point(aes(y = CumulativeVariance), color = "red", size = 2) +
  geom_vline(xintercept = n_components, linetype = "dashed", color = "darkgreen") +
  labs(title = "Scree Plot for PCA",
       subtitle = sprintf("Vertical dashed line indicates %d components (>= %.0f%% variance explained)", n_components, target_threshold*100),
       x = "Principal Component",
       y = "Proportion of Variance Explained") +
  theme_minimal()

# Display the plot
print(scree_plot)

test_numeric <- testNoFeatureEngineering %>%
  select(-label) %>%
  select(where(is.numeric))

# For the training set
pca_train <- as.data.frame(pca_result$x[, 1:183])
pca_train$label <- trainNoFeatureEngineering$label
pca_train$label <- factor(pca_train$label, levels = c("Drowsy", "Natural"))

# For the test set:
scaled_test <- scale(test_numeric, center = pca_result$center, scale = pca_result$scale)
pca_test_full <- as.data.frame(as.matrix(scaled_test) %*% pca_result$rotation)
pca_test <- pca_test_full[, 1:183]  # subset to the first 183 principal components
pca_test$label <- testNoFeatureEngineering$label
pca_test$label <- factor(pca_test$label, levels = c("Drowsy", "Natural"))


# Set up parallel processing
num_cores <- parallel::detectCores() - 1  # reserve one core for OS tasks
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Define cross-validation method for training
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# Train an SVM model.
# Here, we're using the SVM with a radial kernel ("svmRadial"). 
# You can adjust tuning parameters as needed.
svm_model <- train(label ~ ., 
                   data = pca_train, 
                   method = "svmRadial", 
                   trControl = train_control,
                   preProcess = NULL)

# Stop the parallel cluster after training
stopCluster(cl)

# Predict on the test set
predictions <- predict(svm_model, newdata = pca_test)

# Evaluate the model using a confusion matrix
conf_matrix <- confusionMatrix(predictions, pca_test$label)
print(conf_matrix)
print(svm_model$bestTune)