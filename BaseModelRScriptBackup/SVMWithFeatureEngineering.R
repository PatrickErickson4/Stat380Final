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
# Added for ROC analysis
library(pROC)

set.seed(100)

# Load training and testing data
train <- read_csv("Stat380Final/Prepped_Data/trainWithFeatureEngineering.csv", show_col_types = FALSE)
test <- read_csv("Stat380Final/Prepped_Data/testWithFeatureEngineering.csv", show_col_types = FALSE)

# Ensure that the target variable is a factor.
# If necessary, reorder levels so that "Drowsy" is the positive class.
train$label <- factor(train$label, levels = c("Drowsy", "Natural"))
test$label  <- factor(test$label,  levels = c("Drowsy", "Natural"))

# Set up parallel processing
num_cores <- parallel::detectCores() - 1  
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Define cross-validation method for training.
# Note: We add classProbs = TRUE and summaryFunction for ROC
train_control <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train an SVM model.
# The additional argument prob.model = TRUE ensures that ksvm computes probabilities.
svm_model <- train(
  label ~ ., 
  data = train, 
  method = "svmRadial", 
  trControl = train_control,
  preProcess = NULL,
  metric = "ROC",   # Optimize model using ROC
  prob.model = TRUE # Ensure probability estimates are computed
)

# Stop the parallel cluster after training
stopCluster(cl)

# Display model results
print(svm_model$results)
print(svm_model$bestTune)

###########################################
# 1. Save the model and show how to call it later
###########################################
saveRDS(svm_model, file = "Stat380Final/Base_Models/SVMModelWithFeatureEngineering.rds")
cat("Model saved as 'Stat380Final/Base_Models/SVMModelWithFeatureEngineering.rds'. To load it later, use:\n")
cat("loaded_model <- readRDS('Stat380Final/Base_Models/SVMModelWithFeatureEngineering.rds')\n\n")

###########################################
# 2. Save bestTune hyperparameters as CSV
#    (include a column 'model' with the label "SVMWithFeatureEngineering")
###########################################
best_tune_df <- cbind(model = "SVMWithFeatureEngineering", svm_model$bestTune)
write.csv(best_tune_df, "Stat380Final/Base_Models_Data/SVM/BestHyperparams_SVMWithFeatureEngineering.csv", row.names = FALSE)
cat("BestTune hyperparameters saved as 'Stat380Final/Base_Models_Data/SVM/BestHyperparams_SVMWithFeatureEngineering.csv'.\n\n")

###########################################
# 3. Make predictions and create confusion matrix CSV with TP, FP, FN, TN
###########################################
# Predict on the test set (for class labels)
predictions <- predict(svm_model, newdata = test)

# Compute the confusion matrix
conf_matrix <- confusionMatrix(predictions, test$label)
print(conf_matrix)

# Extract confusion matrix table; assumes binary classification.
# Here we define the first level as the positive class ("Drowsy").
levels_labels <- levels(test$label)
if(length(levels_labels) == 2) {
  positive_class <- levels_labels[1]  # "Drowsy"
  negative_class <- levels_labels[2]  # "Natural"
  
  cm_table <- conf_matrix$table
  TP <- cm_table[positive_class, positive_class]
  FP <- cm_table[positive_class, negative_class]
  FN <- cm_table[negative_class, positive_class]
  TN <- cm_table[negative_class, negative_class]
  
  cm_df <- data.frame(model = "SVMWithFeatureEngineering",
                      TP = TP,
                      FP = FP,
                      FN = FN,
                      TN = TN)
  
  write.csv(cm_df, "Stat380Final/Base_Models_Data/SVM/ConfusionMatrix_SVMWithFeatureEngineering.csv", row.names = FALSE)
  cat("Confusion matrix saved as 'Stat380Final/Base_Models_Data/SVM/ConfusionMatrix_SVMWithFeatureEngineering.csv'.\n\n")
  
  ###########################################
  # 4. Calculate and save F1-score, Precision, Recall, Accuracy, and AUC as CSV
  ###########################################
  accuracy <- conf_matrix$overall["Accuracy"]
  recall <- conf_matrix$byClass["Sensitivity"]
  precision <- conf_matrix$byClass["Pos Pred Value"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Obtain probability predictions for ROC curve (now that we enabled probabilities)
  probs <- predict(svm_model, newdata = test, type = "prob")
  # Use probability for the positive class ("Drowsy")
  roc_obj <- roc(response = test$label, predictor = probs[[positive_class]])
  auc_val <- as.numeric(auc(roc_obj))
  
  metrics_df <- data.frame(model = "SVMWithFeatureEngineering",
                           Accuracy = accuracy,
                           Precision = precision,
                           Recall = recall,
                           FOne = f1_score,
                           AUC = auc_val)
  
  write.csv(metrics_df, "Stat380Final/Base_Models_Data/SVM/Metrics_SVMWithFeatureEngineering.csv", row.names = FALSE)
  cat("Performance metrics saved as 'Stat380Final/Base_Models_Data/SVM/Metrics_SVMWithFeatureEngineering.csv'.\n")
  
  ###########################################
  # 5. Create and save an AUC ROC plot as a PNG
  ###########################################
  # Build data for ROC curve: fpr (1 - specificity) and tpr (sensitivity)
  roc_data <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)
  
  write.csv(roc_data, "Stat380Final/Base_Models_Data/SVM/ROCData_SVMWithFeatureEngineering.csv", row.names = FALSE)
  cat("ROC Data for plot building saved as 'Stat380Final/Base_Models_Data/SVM/ROCData_SVMWithFeatureEngineering.csv'.\n")
  
  # Generate the ROC plot using ggplot2.
  roc_plot <- ggplot(roc_data, aes(x = fpr, y = tpr)) +
    geom_line(color = "blue") +  # ROC curve in blue
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "black") +  # Dotted 45° reference line
    labs(x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)",
         title = "ROC Curve for SVMWithFeatureEngineering") +
    theme_minimal() +
    # Annotate AUC in the top right corner, rounded to 3 significant figures.
    annotate("text", x = 0.75, y = 0.95, 
             label = paste("AUC =", format(round(auc_val, 3), nsmall = 3)), 
             color = "red", size = 5)
  
  # Save the ROC plot as a PNG file.
  ggsave("Stat380Final/Base_Models_Data/SVM/ROC_SVMWithFeatureEngineering.png", 
         plot = roc_plot, width = 7, height = 7, dpi = 300)
  cat("ROC plot saved as 'Stat380Final/Base_Models_Data/SVM/ROC_SVMWithFeatureEngineering.png'.\n")
  
} else {
  cat("Skipping confusion matrix extraction and ROC analysis: classification is not binary.\n")
}
