 # XG boost
library(xgboost)
library(Metrics)
library(dplyr)
library(ggplot2)
library(caret)
library(data.table)

# ---- Load Your Dataset ----
data <- read.csv("C:/Users/bhoomika/Desktop/dessetation/garma on climatic factors/min tempr/GARMA MIN TEPR.csv")

# Replace "target" with your actual target column name
target_col <- "Cases"  # <<< CHANGE THIS TO YOUR ACTUAL TARGET

# Split into features and target
library(data.table)
data <- as.data.table(data)
X <- data[, setdiff(names(data), target_col), with = FALSE]

X <- data[, setdiff(names(data), target_col), with = FALSE]
y <- data[[target_col]]

# Convert to data frame and matrix
X <- as.data.frame(X)
X <- as.matrix(X)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)

# Ensure trainIndex is within bounds of the number of rows in X
if (max(trainIndex) <= nrow(X)) {
  X_train <- X[trainIndex, ]
  y_train <- y[trainIndex]
  X_test <- X[-trainIndex, ]
  y_test <- y[-trainIndex]
} else {
  stop("trainIndex contains indices out of bounds.")
}

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Define model parameters
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  print_every_n = 10,
  early_stopping_rounds = 10
)

# Make predictions for the test set
preds <- predict(xgb_model, dtest)

# ---- Calculate Evaluation Metrics ----

# RMSE (Root Mean Squared Error)
rmse_val <- rmse(y_test, preds)

# MAE (Mean Absolute Error)
mae_val <- mae(y_test, preds)

library(rsq)

# Calculate R-squared manually
ss_res <- sum((y_test - preds)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2_val <- 1 - (ss_res / ss_tot)

cat("R-squared (R²):", round(r2_val, 4), "\n")

# Mean Percentage Squared Error (MPSE)
mpse_val <- mean(((y_test - preds) / y_test)^2) * 100

cat("MPSE:", round(mpse_val, 2), "\n")

cat("RMSE: ", rmse_val, "\n")
cat("MAE: ", mae_val, "\n")
cat("MPSE: ", mpse_val, "\n")

# ---- Forecasting the Next 10 Years ----
future_years <- 10
last_data <- tail(X, 1)  # Last row of your data, assuming this is the starting point for prediction

# Generate future features assuming a linear increase for demonstration purposes (adjust accordingly)
future_X <- matrix(NA, nrow = future_years, ncol = ncol(X))
colnames(future_X) <- colnames(X)

for (i in 1:future_years) {
  future_X[i, ] <- last_data + runif(ncol(X), 0, 0.1)  # Adjust with more realistic trend/projection
}

# Predict future cases
future_dmatrix <- xgb.DMatrix(data = future_X)
future_preds <- predict(xgb_model, future_dmatrix)

# Prepare data for plotting
# Ensure consistent column names across all data frames
observed_df <- data.frame(Year = data$Time, Observed_Cases = y)
predicted_df <- data.frame(Year = data$Time[trainIndex], Predicted_Cases = preds)
forecast_years <- seq(from = max(data$Time), by = 1, length.out = future_years)
forecast_df <- data.frame(Year = forecast_years, Forecasted_Cases = future_preds)

# Combine all three data frames
# Ensure the columns match for rbind
observed_df <- rename(observed_df, Case_Type = Observed_Cases)
predicted_df <- rename(predicted_df, Case_Type = Predicted_Cases)
forecast_df <- rename(forecast_df, Case_Type = Forecasted_Cases)

# Add a new column 'Type' to differentiate the three case types
observed_df$Type <- "Observed"
predicted_df$Type <- "Predicted"
forecast_df$Type <- "Forecasted"

# Combine them into a single data frame
combined_df <- rbind(observed_df, predicted_df, forecast_df)

ggplot(combined_df, aes(x = Year)) +
  geom_line(data = observed_df, aes(y = Case_Type, color = Type), linetype = "solid", linewidth = 1) +
  geom_line(data = predicted_df, aes(y = Case_Type, color = Type), linetype = "dashed", linewidth = 1) +
  geom_line(data = forecast_df, aes(y = Case_Type, color = Type), linetype = "dotted", linewidth = 1) +
  ggtitle("JE cases for next 10 years") +
  xlab("Year") +
  ylab("Cases") +
  theme_minimal() +
  scale_color_manual(values = c("black", "blue", "red")) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  labs(color = "Case Type")

   

  # SVM MODEL
# Load necessary libraries
library(e1071)
library(caret)
library(ggplot2)
library(dplyr)

# Load your data (assuming 'data' is the dataset with 'Cases' as the target variable)
# Assuming 'data' has been loaded, and the target column is 'Cases'

# Split the data into training and testing sets (assuming 'Cases' is the target)
set.seed(123)
trainIndex <- createDataPartition(data$Cases, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Define target (Cases) and predictors (X)
target_col <- "Cases"
X_train <- train_data[, setdiff(names(train_data), target_col), with = FALSE]
y_train <- train_data[[target_col]]
X_test <- test_data[, setdiff(names(test_data), target_col), with = FALSE]
y_test <- test_data[[target_col]]

# Fit the Support Vector Regression model (using Radial basis kernel)
svr_model <- svm(Cases ~ ., data = train_data, 
                 kernel = "radial",           # Radial kernel
                 cost = 1,                    # Cost parameter (C)
                 gamma = 0.1,                 # Gamma parameter
                 epsilon = 0.1)               # Epsilon (the margin of error for SVR)

# Make predictions for the test set
preds <- predict(svr_model, newdata = test_data)

# ---- Evaluate the model ----
rmse_val <- sqrt(mean((y_test - preds)^2))
mae_val <- mean(abs(y_test - preds))
ss_res <- sum((y_test - preds)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2_val <- 1 - (ss_res / ss_tot)
mpse_val <- mean(((y_test - preds) / y_test)^2) * 100

cat("R-squared (R²):", round(r2_val, 4), "\n")
cat("RMSE: ", round(rmse_val, 4), "\n")
cat("MAE: ", round(mae_val, 4), "\n")
cat("MPSE:", round(mpse_val, 2), "\n")

# ---- Forecasting the Next 10 Years ----

# Assuming the features (X) stay constant or follow a trend. If your data has a time component, you may create future values accordingly.

# Define number of years to forecast (for example, 10 years ahead)
future_years <- 10

# Ensure `last_data` is the last row from the training data (as a data frame or matrix)
last_data <- tail(X_train, 1)  # Get the most recent row from training data

# Convert `last_data` into a numeric vector
last_data_vec <- as.numeric(last_data)

# Create an empty matrix for future data predictions
future_X <- matrix(NA, nrow = future_years, ncol = ncol(X_train))
colnames(future_X) <- colnames(X_train)

# Generate future data points by adding random noise using `runif`
for (i in 1:future_years) {
  # Ensure `last_data_vec` is a numeric vector and matches the number of features
  future_X[i, ] <- last_data_vec + runif(ncol(X_train), 0, 0.1)  # Adjust randomness as needed
}

# Convert `future_X` to a data frame (optional)
future_X <- as.data.frame(future_X)

# Make predictions using your trained model (SVM)
future_preds <- predict(svr_model, newdata = future_X)

# Forecast years
forecast_years <- seq(from = max(data$Time), by = 1, length.out = future_years)

# Prepare the forecast data frame
forecast_df <- data.frame(Year = forecast_years, Forecasted_Cases = future_preds)

# Combine observed, predicted, and forecast data for plotting
observed_df <- data.frame(Year = data$Time, Observed_Cases = data$Cases)

# Now, ensure that the lengths of the time index and predictions match
predicted_df <- data.frame(
  Year = data$Time[trainIndex],  # Make sure this corresponds to the train set's time values
  Predicted_Cases = predict(svr_model, newdata = X_train)
)

# Add a new column 'Type' to differentiate the data types
predicted_df$Type <- "Predicted"
observed_df$Type <- "Observed"
forecast_df$Type <- "Forecasted"

# Standardize column names
observed_df <- rename(observed_df, Case_Type = Observed_Cases)
predicted_df <- rename(predicted_df, Case_Type = Predicted_Cases)
forecast_df <- rename(forecast_df, Case_Type = Forecasted_Cases)

# Make sure that the columns in each data frame are in the same order and have the same names
observed_df <- observed_df[, c("Year", "Case_Type", "Type")]
predicted_df <- predicted_df[, c("Year", "Case_Type", "Type")]
forecast_df <- forecast_df[, c("Year", "Case_Type", "Type")]

# Now combine the data frames into one
combined_df <- rbind(observed_df, predicted_df, forecast_df)

# Check if the combination worked
head(combined_df)

# Plotting the results
ggplot(combined_df, aes(x = Year)) +
  geom_line(data = observed_df, aes(y = Case_Type, color = Type), linetype = "solid", size = 1) +
  geom_line(data = predicted_df, aes(y = Case_Type, color = Type), linetype = "dashed", size = 1) +
  geom_line(data = forecast_df, aes(y = Case_Type, color = Type), linetype = "dotted", size = 1) +
  ggtitle("Forecast of Cases for the Next 10 Years using SVR Model") +
  xlab("Year") +
  ylab("Cases") +
  theme_minimal() +
  scale_color_manual(values = c("black", "blue", "red")) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  labs(color = "Case Type")

# GAM MODEL

# Load necessary libraries
install.packages("mgcv")
library(mgcv)  # For Generalized Additive Models (GAM)
library(caret)
library(ggplot2)
library(dplyr)

# Load your data (assuming 'data' is the dataset with 'Cases' as the target variable)
# Assuming 'data' has been loaded, and the target column is 'Cases'

# Split the data into training and testing sets (assuming 'Cases' is the target)
set.seed(123)
trainIndex <- createDataPartition(data$Cases, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Define target (Cases) and predictors (X)
target_col <- "Cases"
X_train <- train_data[, setdiff(names(train_data), target_col), with = FALSE]
y_train <- train_data[[target_col]]
X_test <- test_data[, setdiff(names(test_data), target_col), with = FALSE]
y_test <- test_data[[target_col]]

# Fit the Generalized Additive Model (GAM)
gam_model <- gam(Cases ~ s(Time), data = train_data)  # Replace Predictor1, Predictor2 with your actual predictor variables

# Summary of the GAM model
summary(gam_model)

# Make predictions for the test set
preds <- predict(gam_model, newdata = test_data)

# ---- Evaluate the model ----
rmse_val <- sqrt(mean((y_test - preds)^2))
mae_val <- mean(abs(y_test - preds))
ss_res <- sum((y_test - preds)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
r2_val <- 1 - (ss_res / ss_tot)
mpse_val <- mean(((y_test - preds) / y_test)^2) * 100

cat("R-squared (R²):", round(r2_val, 4), "\n")
cat("RMSE: ", round(rmse_val, 4), "\n")
cat("MAE: ", round(mae_val, 4), "\n")
cat("MPSE:", round(mpse_val, 2), "\n")

# ---- Forecasting the Next 10 Years ----

# Assuming the features (X) stay constant or follow a trend. If your data has a time component, you may create future values accordingly.

# Define number of years to forecast (for example, 10 years ahead)
future_years <- 10

# Create future data with the same structure as X_train
last_data <- tail(X_train, 1)  # Get the most recent row from training data
last_data_vec <- as.numeric(last_data)

# Create an empty matrix for future data predictions
future_X <- matrix(NA, nrow = future_years, ncol = ncol(X_train))
colnames(future_X) <- colnames(X_train)

# Generate future data points by adding random noise using `runif`
for (i in 1:future_years) {
  future_X[i, ] <- last_data_vec + runif(ncol(X_train), 0, 0.1)  # Adjust randomness as needed
}

# Convert `future_X` to a data frame (optional)
future_X <- as.data.frame(future_X)

# Make predictions using your trained GAM model
future_preds <- predict(gam_model, newdata = future_X)

# Forecast years
forecast_years <- seq(from = max(data$Time), by = 1, length.out = future_years)

# Prepare the forecast data frame
forecast_df <- data.frame(Year = forecast_years, Forecasted_Cases = future_preds)

# Combine observed, predicted, and forecast data for plotting
observed_df <- data.frame(Year = data$Time, Observed_Cases = data$Cases)

# Now, ensure that the lengths of the time index and predictions match
predicted_df <- data.frame(
  Year = data$Time[trainIndex],  # Make sure this corresponds to the train set's time values
  Predicted_Cases = predict(gam_model, newdata = X_train)
)

# Add a new column 'Type' to differentiate the data types
predicted_df$Type <- "Predicted"
observed_df$Type <- "Observed"
forecast_df$Type <- "Forecasted"

# Standardize column names
observed_df <- rename(observed_df, Case_Type = Observed_Cases)
predicted_df <- rename(predicted_df, Case_Type = Predicted_Cases)
forecast_df <- rename(forecast_df, Case_Type = Forecasted_Cases)

# Make sure that the columns in each data frame are in the same order and have the same names
observed_df <- observed_df[, c("Year", "Case_Type", "Type")]
predicted_df <- predicted_df[, c("Year", "Case_Type", "Type")]
forecast_df <- forecast_df[, c("Year", "Case_Type", "Type")]

# Now combine the data frames into one
combined_df <- rbind(observed_df, predicted_df, forecast_df)

# Check if the combination worked
head(combined_df)

# Plotting the results
ggplot(combined_df, aes(x = Year)) +
  geom_line(data = observed_df, aes(y = Case_Type, color = Type), linetype = "solid", linewidth = 1) +
  geom_line(data = predicted_df, aes(y = Case_Type, color = Type), linetype = "dashed", linewidth = 1) +
  geom_line(data = forecast_df, aes(y = Case_Type, color = Type), linetype = "dotted", linewidth = 1) +
  ggtitle("Forecast of Cases for the Next 10 Years using GAM Model") +
  xlab("Year") +
  ylab("Cases") +
  theme_minimal() +
  scale_color_manual(values = c("black", "blue", "red")) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  labs(color = "Case Type")

#BPNN


# Install required packages
install.packages("neuralnet")
install.packages("caret")
install.packages("Metrics")
install.packages("dplyr")
install.packages("ggplot2")

# Load libraries
library(neuralnet)
library(caret)
library(Metrics)
library(dplyr)
library(ggplot2)
library(data.table)

# Load your dataset
data <- read.csv("C:/Users/bhoomika/Desktop/dessetation/garma on climatic factors/min tempr/GARMA MIN TEPR.csv")

# Make sure 'Cases' is the target column
target_col <- "Cases"

# Normalize the data (important for neural networks)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data_norm <- as.data.frame(lapply(data[, -1], normalize))  # Exclude "Time" if it's the first column
data_norm$Cases <- normalize(data$Cases)  # Normalize target too

# Split into training and test sets
set.seed(123)
trainIndex <- createDataPartition(data_norm$Cases, p = 0.7, list = FALSE)
train_data <- data_norm[trainIndex, ]
test_data <- data_norm[-trainIndex, ]

# Define the formula for neuralnet
predictor_names <- setdiff(names(train_data), target_col)
formula <- as.formula(paste("Cases ~", paste(predictor_names, collapse = " + ")))

# Train the BPNN model
set.seed(123)
bpnn_model <- neuralnet(formula, 
                        data = train_data, 
                        hidden = c(5, 3),  # 2 hidden layers with 5 and 3 neurons
                        linear.output = TRUE)

# Plot the network
plot(bpnn_model)

# Make predictions on test data
predictions <- compute(bpnn_model, test_data[, predictor_names])$net.result

# Denormalize predictions if needed (optional)
# You can skip this if you just care about relative error
actual <- data$Cases[-trainIndex]
preds <- predictions * (max(data$Cases) - min(data$Cases)) + min(data$Cases)

# Evaluation Metrics
rmse_val <- rmse(actual, preds)
mae_val <- mae(actual, preds)
r2_val <- 1 - sum((actual - preds)^2) / sum((actual - mean(actual))^2)
mpse_val <- mean(((actual - preds) / actual)^2) * 100

cat("R-squared (R²):", round(r2_val, 4), "\n")
cat("RMSE: ", round(rmse_val, 4), "\n")
cat("MAE: ", round(mae_val, 4), "\n")
cat("MPSE:", round(mpse_val, 2), "\n")

# ---- Forecast next 10 years ----

# Get the last row from full normalized dataset
last_data <- tail(data_norm[, predictor_names], 1)
future_years <- 10
future_data <- matrix(NA, nrow = future_years, ncol = ncol(last_data))
colnames(future_data) <- colnames(last_data)

# Create synthetic future data with slight random increases (you can update logic)
for (i in 1:future_years) {
  future_data[i, ] <- as.numeric(last_data) + runif(ncol(last_data), 0, 0.01)
}
future_data <- as.data.frame(future_data)

# Predict future values
future_preds_norm <- compute(bpnn_model, future_data)$net.result
future_preds <- future_preds_norm * (max(data$Cases) - min(data$Cases)) + min(data$Cases)

# Create forecast year labels
forecast_years <- seq(from = max(data$Time) + 1, by = 1, length.out = future_years)

# Create data frames for plotting
observed_df <- data.frame(Year = data$Time, Case_Type = data$Cases, Type = "Observed")
predicted_df <- data.frame(Year = data$Time[-trainIndex], Case_Type = preds, Type = "Predicted")
forecast_df <- data.frame(Year = forecast_years, Case_Type = future_preds, Type = "Forecasted")

# Combine all
combined_df <- rbind(observed_df, predicted_df, forecast_df)

# Plot
ggplot(combined_df, aes(x = Year, y = Case_Type, color = Type)) +
  geom_line(aes(linetype = Type), linewidth = 1) +
  scale_color_manual(values = c("Observed" = "black", "Predicted" = "blue", "Forecasted" = "red")) +
  scale_linetype_manual(values = c("Observed" = "solid", "Predicted" = "dashed", "Forecasted" = "dotted")) +
  ggtitle("BPNN Forecast of JE Cases (Next 10 Years)") +
  xlab("Year") + ylab("Cases") +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")

#RANDOM FOREST

# ---- Load Required Libraries ----
install.packages("randomForest")
library(randomForest)
library(Metrics)
library(dplyr)
library(ggplot2)
library(caret)
library(data.table)

# ---- Load Your Dataset ----
data <- read.csv("C:/Users/bhoomika/Desktop/dessetation/garma on climatic factors/min tempr/GARMA MIN TEPR.csv")

# Replace "Cases" with your actual target column name
target_col <- "Cases"

# Convert data to data.table
data <- as.data.table(data)
X <- data[, setdiff(names(data), target_col), with = FALSE]
y <- data[[target_col]]

# Convert to data frame
X <- as.data.frame(X)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)

X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# ---- Train Random Forest Model ----
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500, mtry = floor(sqrt(ncol(X_train))))

# ---- Predictions ----
preds <- predict(rf_model, newdata = X_test)

# ---- Evaluation Metrics ----
rmse_val <- rmse(y_test, preds)
mae_val <- mae(y_test, preds)
r2_val <- 1 - sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2)
mpse_val <- mean(((y_test - preds) / y_test)^2) * 100

cat("R-squared (R²):", round(r2_val, 4), "\n")
cat("RMSE: ", round(rmse_val, 4), "\n")
cat("MAE: ", round(mae_val, 4), "\n")
cat("MPSE: ", round(mpse_val, 2), "\n")

# ---- Forecasting the Next 10 Years ----
future_years <- 10
last_data <- tail(X, 1)

future_X <- matrix(NA, nrow = future_years, ncol = ncol(X))
colnames(future_X) <- colnames(X)

for (i in 1:future_years) {
  future_X[i, ] <- as.numeric(last_data) + runif(ncol(X), 0, 0.1)
}
future_X <- as.data.frame(future_X)

future_preds <- predict(rf_model, newdata = future_X)

# ---- Prepare Data for Plotting ----
observed_df <- data.frame(Year = data$Time, Case_Type = y, Type = "Observed")
predicted_df <- data.frame(Year = data$Time[-trainIndex], Case_Type = preds, Type = "Predicted")
forecast_years <- seq(from = max(data$Time) + 1, by = 1, length.out = future_years)
forecast_df <- data.frame(Year = forecast_years, Case_Type = future_preds, Type = "Forecasted")

combined_df <- rbind(observed_df, predicted_df, forecast_df)

# ---- Plot ----
ggplot(combined_df, aes(x = Year)) +
  geom_line(data = observed_df, aes(y = Case_Type, color = Type), linetype = "solid", linewidth = 1) +
  geom_line(data = predicted_df, aes(y = Case_Type, color = Type), linetype = "dashed", linewidth = 1) +
  geom_line(data = forecast_df, aes(y = Case_Type, color = Type), linetype = "dotted", linewidth = 1) +
  ggtitle("JE cases for next 10 years (Random Forest)") +
  xlab("Year") +
  ylab("Cases") +
  theme_minimal() +
  scale_color_manual(values = c("black", "blue", "red")) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  labs(color = "Case Type")

#DECISION TREE
# ---- Load Required Libraries ----
library(rpart)
library(Metrics)
library(dplyr)
library(ggplot2)
library(caret)
library(data.table)

# ---- Load Your Dataset ----
data <- read.csv("C:/Users/bhoomika/Desktop/dessetation/garma on climatic factors/TEMPERATURE.csv")

# Replace "Cases" with your actual target column name
target_col <- "Cases"

# Convert data to data.table
data <- as.data.table(data)
X <- data[, setdiff(names(data), target_col), with = FALSE]
y <- data[[target_col]]

# Convert to data frame
X <- as.data.frame(X)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)

X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# ---- Train Decision Tree Model ----
train_data <- cbind(X_train, Cases = y_train)
dt_model <- rpart(Cases ~ ., data = train_data, method = "anova")

# ---- Predictions ----
preds <- predict(dt_model, newdata = X_test)

# ---- Evaluation Metrics ----
rmse_val <- rmse(y_test, preds)
mae_val <- mae(y_test, preds)
r2_val <- 1 - sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2)
mpse_val <- mean(((y_test - preds) / y_test)^2) * 100

cat("R-squared (R²):", round(r2_val, 4), "\n")
cat("RMSE: ", round(rmse_val, 4), "\n")
cat("MAE: ", round(mae_val, 4), "\n")
cat("MPSE: ", round(mpse_val, 2), "\n")

# ---- Forecasting the Next 10 Years ----
future_years <- 10
last_data <- tail(X, 1)

future_X <- matrix(NA, nrow = future_years, ncol = ncol(X))
colnames(future_X) <- colnames(X)

for (i in 1:future_years) {
  future_X[i, ] <- as.numeric(last_data) + runif(ncol(X), 0, 0.1)
}
future_X <- as.data.frame(future_X)

future_preds <- predict(dt_model, newdata = future_X)

# ---- Prepare Data for Plotting ----
observed_df <- data.frame(Year = data$Time, Case_Type = y, Type = "Observed")
predicted_df <- data.frame(Year = data$Time[-trainIndex], Case_Type = preds, Type = "Predicted")
forecast_years <- seq(from = max(data$Time) + 1, by = 1, length.out = future_years)
forecast_df <- data.frame(Year = forecast_years, Case_Type = future_preds, Type = "Forecasted")

combined_df <- rbind(observed_df, predicted_df, forecast_df)

# ---- Plot ----
ggplot(combined_df, aes(x = Year)) +
  geom_line(data = observed_df, aes(y = Case_Type, color = Type), linetype = "solid", linewidth = 1) +
  geom_line(data = predicted_df, aes(y = Case_Type, color = Type), linetype = "dashed", linewidth = 1) +
  geom_line(data = forecast_df, aes(y = Case_Type, color = Type), linetype = "dotted", linewidth = 1) +
  ggtitle("JE cases for next 10 years (Decision Tree)") +
  xlab("Year") +
  ylab("Cases") +
  theme_minimal() +
  scale_color_manual(values = c("black", "blue", "red")) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  labs(color = "Case Type")

XGboost

# ---- Load Required Libraries ----
library(xgboost)
library(caret)
library(dplyr)
library(data.table)
library(Matrix)
library(ggplot2)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- View and Inspect ----
str(data)        # Check structure
head(data)

# ---- Define Target Column for Classification ----
# You may need to convert 'Cases' into categorical classes if not already (e.g., Low/High)
# For example: Convert numeric Cases to binary: 0 = Low, 1 = High
# Adjust threshold according to your context

threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, 1, 0)  # Binary classification

# Remove the original 'Cases' if needed
data$Cases <- NULL

# ---- Split Features and Target ----
target_col <- "Target"
X <- data[, setdiff(names(data), target_col)]
y <- data[[target_col]]

# Convert to matrix for XGBoost
X_matrix <- as.matrix(X)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X_matrix[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X_matrix[-trainIndex, ]
y_test <- y[-trainIndex]
X_train <- as.matrix(X_train)
y_train <- as.numeric(y_train)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

X_test <- as.matrix(X_test)
y_test <- as.numeric(y_test)
dtest <- xgb.DMatrix(data = X_test, label = y_test)


# ---- Set Parameters for Classification ----
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  eta = 0.1,
  max_depth = 6
)

# ---- Train the Model ----
xgb_clf <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  print_every_n = 10,
  early_stopping_rounds = 10
)

# ---- Make Predictions ----
pred_probs <- predict(xgb_clf, X_test)
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# ---- Evaluation ----
conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(y_test))
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

# ---- Optional: Feature Importance Plot ----
importance_matrix <- xgb.importance(feature_names = colnames(X), model = xgb_clf)
xgb.plot.importance(importance_matrix, top_n = 10)

# ---- Additional Metrics (Regression-style on probabilities) ----
# Install Metrics package if not installed
if (!require("Metrics")) install.packages("Metrics", dependencies = TRUE)
library(Metrics)

# Calculate metrics on predicted probabilities vs actual binary labels
rmse_val <- rmse(y_test, pred_probs)
mae_val <- mae(y_test, pred_probs)
r2_val <- 1 - sum((y_test - pred_probs)^2) / sum((y_test - mean(y_test))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

 adaboost

# ---- Load Required Libraries ----
library(caret)
library(dplyr)
library(ggplot2)
library(adabag)
library(Metrics)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- View and Inspect ----
str(data)
head(data)

# ---- Create Binary Target from 'Cases' ----
threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, 1, 0)
data$Cases <- NULL

# ---- Convert Target to Factor (required for AdaBoost) ----
data$Target <- as.factor(data$Target)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(data$Target, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# ---- Train AdaBoost Model ----
ada_model <- boosting(Target ~ ., data = train_data, boos = TRUE, mfinal = 100)

# ---- Make Predictions ----
pred <- predict.boosting(ada_model, newdata = test_data)

# ---- Evaluation ----
conf_mat <- confusionMatrix(as.factor(pred$class), test_data$Target)
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

# ---- Additional Metrics (RMSE, MAE, R2 on Probabilities) ----
probabilities <- pred$prob[, 2]  # Probability for class "1"
true_labels <- as.numeric(as.character(test_data$Target))

rmse_val <- rmse(true_labels, probabilities)
mae_val <- mae(true_labels, probabilities)
r2_val <- 1 - sum((true_labels - probabilities)^2) / sum((true_labels - mean(true_labels))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

#
modelLookup()

   random forest
# ---- Load Required Libraries ----
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(Metrics)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- View and Inspect ----
str(data)
head(data)

# ---- Create Binary Target from 'Cases' ----
threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, 1, 0)
data$Cases <- NULL

# ---- Convert Target to Factor (required for classification) ----
data$Target <- as.factor(data$Target)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(data$Target, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# ---- Train Random Forest Model ----
rf_model <- randomForest(Target ~ ., data = train_data, ntree = 100, importance = TRUE)

# ---- Make Predictions ----
pred_class <- predict(rf_model, newdata = test_data, type = "response")
pred_prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2]  # Prob for class "1"

# ---- Evaluation ----
conf_mat <- confusionMatrix(pred_class, test_data$Target)
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

# ---- Additional Metrics (RMSE, MAE, R² on Probabilities) ----
true_labels <- as.numeric(as.character(test_data$Target))

rmse_val <- rmse(true_labels, pred_prob)
mae_val <- mae(true_labels, pred_prob)
r2_val <- 1 - sum((true_labels - pred_prob)^2) / sum((true_labels - mean(true_labels))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

# ---- Optional: Variable Importance Plot ----
varImpPlot(rf_model, main = "Random Forest - Feature Importance")

****************** decision tree*********************
  
# ---- Load Required Libraries ----
library(caret)
library(dplyr)
library(ggplot2)
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
library(Metrics)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- View and Inspect ----
str(data)
head(data)

# ---- Create Binary Target from 'Cases' ----
threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, 1, 0)
data$Cases <- NULL

# ---- Convert Target to Factor (required for classification) ----
data$Target <- as.factor(data$Target)

# ---- Train-Test Split ----
set.seed(123)
trainIndex <- createDataPartition(data$Target, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# ---- Train Decision Tree Model ----
dt_model <- rpart(Target ~ ., data = train_data, method = "class", parms = list(split = "gini"))

# ---- Plot the Tree ----
rpart.plot(dt_model, main = "Decision Tree", type = 3, extra = 101)

# ---- Make Predictions ----
pred_class <- predict(dt_model, newdata = test_data, type = "class")
pred_prob <- predict(dt_model, newdata = test_data, type = "prob")[, 2]

# ---- Evaluation ----
conf_mat <- confusionMatrix(pred_class, test_data$Target)
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

# ---- Additional Metrics (RMSE, MAE, R² on Probabilities) ----
true_labels <- as.numeric(as.character(test_data$Target))

rmse_val <- rmse(true_labels, pred_prob)
mae_val <- mae(true_labels, pred_prob)
r2_val <- 1 - sum((true_labels - pred_prob)^2) / sum((true_labels - mean(true_labels))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

knn

# ---- Load Required Libraries ----
library(caret)
library(dplyr)
library(Metrics)
library(ggplot2)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- View Structure ----
str(data)
head(data)

# ---- Create Binary Target ----
threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, 1, 0)
data$Cases <- NULL
data$Target <- as.factor(data$Target)

# Ensure feature_data is a data frame
feature_data <- data[, setdiff(names(data), "Target")]
feature_data <- as.data.frame(feature_data)

# Step 1: Make sure feature_data is a data frame
feature_data <- as.data.frame(data[, setdiff(names(data), "Target")])

# Step 2: Filter out columns with zero or NA standard deviation
filtered_features <- feature_data[, sapply(feature_data, function(x) {
  is.numeric(x) && !all(is.na(x)) && sd(x, na.rm = TRUE) > 0
})]

# Step 3: Force filtered_features to stay a data frame
filtered_features <- as.data.frame(filtered_features)

# Step 4: Preprocessing (Center and Scale)
pre_proc <- preProcess(filtered_features, method = c("center", "scale"))
scaled_features <- predict(pre_proc, filtered_features)

# ---- Combine with Target ----
processed_data <- cbind(scaled_features, Target = target_data)

# ---- Train-Test Split ----
set.seed(123)
train_index <- createDataPartition(processed_data$Target, p = 0.7, list = FALSE)
train_data <- processed_data[train_index, ]
test_data <- processed_data[-train_index, ]

# ---- Train KNN Model ----
ctrl <- trainControl(method = "cv", number = 5)
knn_model <- train(Target ~ ., data = train_data, method = "knn", trControl = ctrl, tuneLength = 10)

# ---- Predict on Test Set ----
pred_class <- predict(knn_model, newdata = test_data)
pred_prob <- predict(knn_model, newdata = test_data, type = "prob")[, 2]

# ---- Evaluation ----
conf_mat <- confusionMatrix(pred_class, test_data$Target)
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

# ---- Metrics: RMSE, MAE, R-squared ----
true_labels <- as.numeric(as.character(test_data$Target))

rmse_val <- rmse(true_labels, pred_prob)
mae_val <- mae(true_labels, pred_prob)
r2_val <- 1 - sum((true_labels - pred_prob)^2) / sum((true_labels - mean(true_labels))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

SVM
# ---- Load Required Libraries ----
library(caret)
library(dplyr)
library(Metrics)
library(ggplot2)

# ---- Load Dataset ----
data <- read.csv("C:/Users/bhoomika/.spss/Downloads/ARMA-cases and time.csv")

# ---- Create Binary Target ----
threshold <- median(data$Cases, na.rm = TRUE)
data$Target <- ifelse(data$Cases > threshold, "High", "Low")  # Valid factor levels
data$Cases <- NULL
data$Target <- as.factor(make.names(data$Target))  # Ensure safe factor names for caret

# ---- Separate Features and Target ----
feature_data <- data[, setdiff(names(data), "Target")]
target_data <- data$Target

# ---- Handle Single Column Edge Case ----
if (is.null(ncol(feature_data))) {
  feature_data <- data.frame(feature_data)  # Convert to data frame
  names(feature_data) <- "Feature1"         # Name the column
}

# ---- Remove Columns with Zero or NA Std Dev ----
filtered_features <- feature_data[, sapply(feature_data, function(x) {
  if (is.numeric(x)) {
    sd_x <- sd(x, na.rm = TRUE)
    !is.na(sd_x) && sd_x > 0
  } else {
    FALSE
  }
}), drop = FALSE]

# ---- Preprocess (Center & Scale) ----
filtered_features <- as.data.frame(filtered_features)
pre_proc <- preProcess(filtered_features, method = c("center", "scale"))
scaled_features <- predict(pre_proc, filtered_features)

# ---- Combine with Target ----
processed_data <- cbind(scaled_features, Target = target_data)

# ---- Train-Test Split ----
set.seed(123)
train_index <- createDataPartition(processed_data$Target, p = 0.7, list = FALSE)
train_data <- processed_data[train_index, ]
test_data <- processed_data[-train_index, ]

# ---- Train SVM Model ----
ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

svm_model <- caret::train(
  Target ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 10,
  metric = "ROC"
)

# ---- Predict on Test Set ----
pred_class <- predict(svm_model, newdata = test_data)
pred_prob <- predict(svm_model, newdata = test_data, type = "prob")[, "High"]

# ---- Evaluation ----
# Ensure factor levels match
pred_class <- factor(pred_class, levels = levels(test_data$Target))
conf_mat <- confusionMatrix(pred_class, test_data$Target)
print(conf_mat)

# ---- Metrics: Accuracy, RMSE, MAE, R2 ----
accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy:", round(accuracy, 4), "\n")

true_labels <- ifelse(test_data$Target == "High", 1, 0)
rmse_val <- rmse(true_labels, pred_prob)
mae_val <- mae(true_labels, pred_prob)
r2_val <- 1 - sum((true_labels - pred_prob)^2) / sum((true_labels - mean(true_labels))^2)

cat("RMSE:", round(rmse_val, 4), "\n")
cat("MAE:", round(mae_val, 4), "\n")
cat("R-squared:", round(r2_val, 4), "\n")

