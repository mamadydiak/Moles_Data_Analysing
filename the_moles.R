library(caret)
library(skimr)
library(RANN)
library(randomForest)
library(klaR)

data <- read.csv("mole2_dataset.csv")

str(data)

data$image_id <- as.factor(data$image_id)
data$sex <- as.factor(data$sex)
data$diagnosis <- as.factor(data$diagnosis)

str(data)

# Step 2: Descriptive Statistics
skim_to_wide(data)
## comment yapD1lacak
# 
# Factor Columns:
#   image_id:
# 
#   No missing values
# All unique values (400 unique image IDs)
# Comment: Each image ID is unique, suggesting that it serves as a primary identifier for individual images.
# sex:
# 
#   No missing values
# Three unique values: Male (210), Female (189), Emp (1)
# Comment: The dataset includes more male images than female images. There is also one instance labeled as "emp."
# diagnosis:
# 
#   No missing values
# Four unique values: Mel (195), Nev (142), Seb (58), Bas (5)
# Comment: The dataset includes images with different skin diagnoses, with melanoma (mel) being the most frequent, followed by nevus (nev), seborrheic keratosis (seb), and basal cell carcinoma (bas).
# Numeric Columns:
#   X (Image Index):
# 
#   No missing values
# Ranges from 0 to 399
# Comment: The image index appears to be a sequential identifier.
# pixels_x (Image Width):
# 
#   No missing values
# Ranges from 640 to 6000
# Comment: The width of the images varies, with a considerable range of values.
# pixels_y (Image Height):
# 
#   No missing values
# Ranges from 480 to 4000
# Comment: Similar to width, the height of the images shows considerable variation.
# clin_size_long_diam_mm (Clinical Size - Long Diameter in mm):
# 
#   264 missing values (34% missing)
# Ranges from 1.6 to 20.1
# Comment: There is a significant amount of missing clinical size data.
# full_img_size (Full Image Size):
# 
#   No missing values
# Ranges from 24,653 to 1,907,816
# Comment: The full image size varies widely, with some images being significantly larger than others.
# age_approx (Age):
# 
#   1 missing value (0.25% missing)
# Ranges from 10 to 85
# Comment: Age values are available for the majority of the dataset.
# non_overlapping_rate:
# 
#   No missing values
# Ranges from 0 to 200
# Comment: The non-overlapping rate has a wide range, indicating the extent of non-overlapping regions in the images.
# corners:
# 
#   No missing values
# Ranges from 0 to 2280
# Comment: The number of corners in the images varies, with some images having a large number of corners.
# red, green, blue (Color Channels):
# 
#   20 missing values (5% missing) for each color channel
# Ranges from 1.86 to 235 (red), 1.86 to 223 (green), 1.86 to 214 (blue)
# Comment: The color channels exhibit variability in their values.
# These descriptive statistics offer insights into the characteristics of each column in the dataset, including the distribution of factor levels and the range of numeric values.


# Step 3: Impute Missing Values using k-NN
preproc_model_impute <- preProcess(data, method = "knnImpute")
data_imputed <- predict(preproc_model_impute, data)
## while modelling data to get a better result we imputed the missing values using knn algorithm.
## K-NN imputation is based on the similarity of instances in the dataset. It replaces missing values with values from neighboring instances,
#considering a specified number of nearest neighbors (k).
## comment yapD1lacak

# Step 4: Variable Transformations (Range 0-1)
preproc_model_range <- preProcess(data_imputed, method = "range", range = c(0, 1))
data_transformed <- predict(preproc_model_range, data_imputed)

# Step 1: Data Splitting
## X ve ID tutmalD1 mD1yD1z??? kaldD1rdD1ktan sonra accuracy dC<EtC<.
data_transformed <- data_transformed[,3:14]

set.seed(123)  # For reproducibility
train_index <- createDataPartition(data_transformed$diagnosis, p = 0.8, list = FALSE)
train_data <- data_transformed[train_index, ]
test_data <- data_transformed[-train_index, ]

# Extract numeric columns only
numeric_columns <- sapply(train_data, is.numeric)
numeric_data <- train_data[, numeric_columns]


##str(x) ## checking numeric variables

# Step 5: Feature Influence Visualization
featurePlot(x = numeric_data, 
            y = train_data$diagnosis, 
            plot = "boxplot")

# If the boxplots show significant differences between each other, it may indicate that the corresponding numerical 
# feature has a strong ability to distinguish between classes.
# 
# If the boxplots are very similar to each other, it may suggest that the feature is weaker in distinguishing a specific class.
# The mean and the placement of boxes are glaringly different for blue, green and non overlapping rate. blue, green and non 
#overlapping rate are going to be a significant predictor of diagnosis.




# Step 6: k-Nearest Neighbors (k-NN)
#a
knn_model <- train(diagnosis ~ ., data = train_data, method = "knn",
                   trControl = trainControl(method = "cv", number = 10)
                   )
#b
test_predictions_knn <- predict(knn_model, newdata = test_data)

# Check lengths of the predicted values and test data labels
# length(test_predictions_knn)
# length(test_data$diagnosis)

#c
confusionMatrix(test_predictions_knn, test_data$diagnosis)


# # Step 7: Random Forest
# a. Train the Random Forest model

rf_model <- train(
  diagnosis ~ ., 
  data = train_data, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5)
  )

# b. Make predictions on the test data
test_predictions_rf <- predict(rf_model, newdata = test_data)

# c. Evaluate the Random Forest model
confusionMatrix(test_predictions_rf, test_data$diagnosis)


# Step 8: NaC/ve Bayes Classification
#a
nb_model <- train(diagnosis ~ ., data = train_data, method = "nb",
                  trControl = trainControl(method = "cv", number = 5)
                  )
#b
test_predictions_nb <- predict(nb_model, newdata = test_data)
#c
confusionMatrix(test_predictions_nb, test_data$diagnosis)

# Step 9: Compare and make more and more comments about


# General Observations:
#   Accuracy:Random Forest has the highest accuracy among the three models (73.42%), followed by k-Nearest Neighbors (63.29%) and Naive Bayes (62.03%).

#  Kappa:Random Forest also has the highest Kappa coefficient at 0.5513. The other models have lower Kappa values.

#   Class-Specific Metrics:
#
#   Each model has  different  levels of sensitivity and specificity. For example, Random Forest shows high sensitivity
#   for melanoma and seborrheic keratosis, indicating good performance in identifying these classes.

# Balanced Accuracy: Random Forest has the highest balanced accuracy.

# Overall Comparison:
#
#   Random Forest generally outperforms the other models in terms of accuracy and class-specific performance.
# In summary,based on the results  Random Forest is the most effective model among the three for the given classification task.





