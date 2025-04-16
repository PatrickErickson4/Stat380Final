# Load necessary libraries
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
library(EBImage)
library(xgboost)  # For the xgbTree method
# Added for ROC analysis
library(pROC)

set.seed(100)

# Load training and testing data
pca_train <- read_csv("Stat380Final/Prepped_Data/trainNoFeatureEngineeringPCA.csv", show_col_types = FALSE)
pca_test <- read_csv("Stat380Final/Prepped_Data/testNoFeatureEngineeringPCA.csv", show_col_types = FALSE)

# Ensure that the target variable is a factor.
# Reorder levels so that "Drowsy" is the positive class.
pca_train$label <- factor(pca_train$label, levels = c("Drowsy", "Natural"))
pca_test$label  <- factor(pca_test$label,  levels = c("Drowsy", "Natural"))

# Set up parallel processing
num_cores <- parallel::detectCores() - 1  
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Define cross-validation method for training.
# We add classProbs = TRUE and summaryFunction = twoClassSummary to get AUC and probability estimates.
train_control <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train an XGBoost model using caret's xgbTree method
xgb_model <- train(
  label ~ ., 
  data = pca_train, 
  method = "xgbTree", 
  trControl = train_control,
  preProcess = NULL,
  metric = "ROC"  # Optimize model using ROC
)

# Stop the parallel cluster after training
stopCluster(cl)

# Print the results and best tuning parameters
print(xgb_model$results)
print(xgb_model$bestTune)

###########################################
# 1. Save the model and show how to call it later
###########################################
saveRDS(xgb_model, file = "Stat380Final/Base_Models/XGBoostNoFeatureEngineeringPCA.rds")
cat("Model saved as 'Stat380Final/Base_Models/XGBoostNoFeatureEngineeringPCA.rds'. To load it later, use:\n")
cat("loaded_model <- readRDS('Stat380Final/Base_Models/XGBoostNoFeatureEngineeringPCA.rds')\n\n")

###########################################
# 2. Save bestTune hyperparameters as CSV
#    (include a column 'model' with the label "XGBoostNoFeatureEngineeringPCA")
###########################################
best_tune_df <- cbind(model = "XGBoostNoFeatureEngineeringPCA", xgb_model$bestTune)
write.csv(best_tune_df, "Stat380Final/Base_Models_Data/XGBoost/BestHyperparams_XGBoostNoFeatureEngineeringPCA.csv", row.names = FALSE)
cat("BestTune hyperparameters saved as 'Stat380Final/Base_Models_Data/XGBoost/BestHyperparams_XGBoostNoFeatureEngineeringPCA.csv'.\n\n")

###########################################
# 3. Make predictions and create confusion matrix CSV with TP, FP, FN, TN
###########################################
# Predict on the test set (for class labels)
predictions <- predict(xgb_model, newdata = pca_test)

# Compute the confusion matrix
conf_matrix <- confusionMatrix(predictions, pca_test$label)
print(conf_matrix)

# Extract confusion matrix table; assumes binary classification.
# Here we define the first level as the positive class ("Drowsy").
levels_labels <- levels(pca_test$label)
if(length(levels_labels) == 2) {
  positive_class <- levels_labels[1]  # "Drowsy"
  negative_class <- levels_labels[2]  # "Natural"
  
  cm_table <- conf_matrix$table
  TP <- cm_table[positive_class, positive_class]
  FP <- cm_table[positive_class, negative_class]
  FN <- cm_table[negative_class, positive_class]
  TN <- cm_table[negative_class, negative_class]
  
  cm_df <- data.frame(model = "XGBoostNoFeatureEngineeringPCA",
                      TP = TP,
                      FP = FP,
                      FN = FN,
                      TN = TN)
  
  write.csv(cm_df, "Stat380Final/Base_Models_Data/XGBoost/ConfusionMatrix_XGBoostNoFeatureEngineeringPCA.csv", row.names = FALSE)
  cat("Confusion matrix saved as 'Stat380Final/Base_Models_Data/XGBoost/ConfusionMatrix_XGBoostNoFeatureEngineeringPCA.csv'.\n\n")
  
  ###########################################
  # 4. Calculate and save F1-score, Precision, Recall, Accuracy, and AUC as CSV
  ###########################################
  accuracy <- conf_matrix$overall["Accuracy"]
  recall <- conf_matrix$byClass["Sensitivity"]
  precision <- conf_matrix$byClass["Pos Pred Value"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Obtain probability predictions for ROC curve (ensure type = "prob")
  probs <- predict(xgb_model, newdata = pca_test, type = "prob")
  # Use probability for the positive class ("Drowsy")
  roc_obj <- roc(response = pca_test$label, predictor = probs[[positive_class]])
  auc_val <- as.numeric(auc(roc_obj))
  
  metrics_df <- data.frame(model = "XGBoostNoFeatureEngineeringPCA",
                           Accuracy = accuracy,
                           Precision = precision,
                           Recall = recall,
                           FOne = f1_score,
                           AUC = auc_val)
  
  write.csv(metrics_df, "Stat380Final/Base_Models_Data/XGBoost/Metrics_XGBoostNoFeatureEngineeringPCA.csv", row.names = FALSE)
  cat("Performance metrics saved as 'Stat380Final/Base_Models_Data/XGBoost/Metrics_XGBoostNoFeatureEngineeringPCA.csv'.\n")
  
  ###########################################
  # 5. Create and save an AUC ROC plot as a PNG
  ###########################################
  # Build data for ROC curve: fpr (1 - specificity) and tpr (sensitivity)
  roc_data <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)
  
  write.csv(roc_data, "Stat380Final/Base_Models_Data/XGBoost/ROCData_XGBoostNoFeatureEngineeringPCA.csv", row.names = FALSE)
  cat("ROC Data for plot building saved as 'Stat380Final/Base_Models_Data/XGBoost/ROCData_XGBoostNoFeatureEngineeringPCA.csv'.\n")
  
  # Generate the ROC plot using ggplot2.
  roc_plot <- ggplot(roc_data, aes(x = fpr, y = tpr)) +
    geom_line(color = "blue") +  # ROC curve in blue
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "black") +  # Dotted 45° reference line
    labs(x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)",
         title = "ROC Curve for XGBoostNoFeatureEngineeringPCA") +
    theme_minimal() +
    # Annotate AUC in the top right corner, rounded to 3 significant figures.
    annotate("text", x = 0.75, y = 0.95, 
             label = paste("AUC =", format(round(auc_val, 3), nsmall = 3)), 
             color = "red", size = 5)
  
  # Save the ROC plot as a PNG file.
  ggsave("Stat380Final/Base_Models_Data/XGBoost/ROC_XGBoostNoFeatureEngineeringPCA.png", 
         plot = roc_plot, width = 7, height = 7, dpi = 300)
  cat("ROC plot saved as 'Stat380Final/Base_Models_Data/XGBoost/ROC_XGBoostNoFeatureEngineeringPCA.png'.\n")
  
} else {
  cat("Skipping confusion matrix extraction and ROC analysis: classification is not binary.\n")
}
