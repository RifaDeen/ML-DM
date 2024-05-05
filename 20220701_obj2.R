# Load the dataset
library(readxl)
rates <- read_excel("ExchangeUSD.xlsx")


################# Input vectors and input/outputt matrices ###########################

library(dplyr)

# Use only the 3rd column (exchange rate data)
data <- rates[, 3]

# Time lagged vectors

l1 <- lag(data,1)
l2 <- lag(data,2)
l3 <- lag(data,3)
l4 <- lag(data,4)

# Create matrices

io1 <- setNames(bind_cols(l1, l3, l4, l2, data), c("lag1","lag3","lag4","lag2", "actual"))
io2 <- setNames(bind_cols(l2, l1, l4, l3, data), c("lag2", "lag1","lag4","lag3","actual"))
io3 <- setNames(bind_cols(l1, l2, l3, l4, data), c("lag1", "lag2", "lag3","lag4", "actual"))
io4 <- setNames(bind_cols(l4, l3, l2, l1, data),
                c("lag4", "lag3", "lag2", "lag1", "actual"))

# Remove NA
io1 <- io1[complete.cases(io1),]
io2 <- io2[complete.cases(io2),]
io3 <- io3[complete.cases(io3),]
io4 <- io4[complete.cases(io4),]


################### normalising #########################


# Function to normalize data
normalize <- function(data) {
  return((data - min(data))/(max(data) - min(data)))
}

io1_norm <- normalize(io1)
io2_norm <- normalize(io2)
io3_norm <- normalize(io3)
io4_norm <- normalize(io4)
head(io2_norm)

################## separate train and test ###############

io1Train <- as.data.frame(io1_norm[1:400, 1:5])
io1Test <- as.data.frame(io1_norm[401:nrow(io1_norm), 1:5])

io2Train <- as.data.frame(io2_norm[1:400, 1:5])
io2Test <- as.data.frame(io2_norm[401:nrow(io2_norm), 1:5])

io3Train <- as.data.frame(io3_norm[1:400, 1:5])
io3Test <- as.data.frame(io3_norm[401:nrow(io3_norm), 1:5])

io4Train <- as.data.frame(io4_norm[1:400, 1:5])
io4Test <- as.data.frame(io4_norm[401:nrow(io4_norm), 1:5])


############### mlp models ########################


library(neuralnet)

# Define MLP models with dropout regularization and different activation functions
set.seed(123)
mlp1 <- neuralnet(actual ~ ., data = io1Train, hidden = c(6), linear.output = TRUE, act.fct = "tanh")
set.seed(123)
mlp2 <- neuralnet(actual ~ ., data = io2Train, hidden = c(8), linear.output = FALSE, act.fct = "tanh")
set.seed(123)
mlp3 <- neuralnet(actual ~ ., data = io3Train, hidden = c(5), linear.output = TRUE, act.fct = "tanh")
set.seed(123)
mlp4 <- neuralnet(actual ~ ., data = io4Train, hidden = c(4), linear.output = FALSE, act.fct = "tanh")

# Plot the MLP models
plot(mlp1)
plot(mlp2)
plot(mlp3)
plot(mlp4)

# Function to train an MLP model with customizable parameters
train_mlp_model <- function(train_data, hidden_layers, activation, linear_output = TRUE) {
  model <- neuralnet(actual ~ ., data = train_data, hidden = hidden_layers, 
                     linear.output = linear_output, act.fct = activation)
  return(model)
}

# Train MLP models with different structures
set.seed(123)
mlp5 <- train_mlp_model(io1Train, c(6, 5), "tanh")
set.seed(123)
mlp6 <- train_mlp_model(io1Train, c(8, 6, 10),"tanh")
set.seed(123)
mlp7 <- train_mlp_model(io2Train, c(12, 8), "logistic")
set.seed(123)
mlp8 <- train_mlp_model(io2Train, c(8, 5, 4), "logistic")
set.seed(123)
mlp9 <- train_mlp_model(io3Train, c(9, 8), "tanh")
set.seed(123)
mlp10 <- train_mlp_model(io3Train, c(6, 4, 10), "tanh")
set.seed(123)
mlp11 <- train_mlp_model(io4Train, c(8, 12), "logistic")
set.seed(123)
mlp12 <- train_mlp_model(io4Train, c(2, 5, 3), "tanh")



# Plot the MLP models
plot(mlp5) 
plot(mlp6)
plot(mlp7)
plot(mlp8)
plot(mlp9)
plot(mlp10)
plot(mlp11)
plot(mlp12)


############# Predictions ###################

# Function to compute predictions using a trained MLP model
compute_predictions <- function(model, test_data) {
  predictions <- predict(model, test_data)
  return (predictions)
}

# Compute predictions for each model

pred1 <- compute_predictions(mlp1, io1Test)
pred2 <- compute_predictions(mlp2, io2Test)
pred3 <- compute_predictions(mlp3, io3Test)
pred4 <- compute_predictions(mlp4, io4Test)
pred5 <- compute_predictions(mlp5, io1Test)
pred6 <- compute_predictions(mlp6, io1Test)
pred7 <- compute_predictions(mlp7, io2Test)
pred8 <- compute_predictions(mlp8, io2Test)
pred9 <- compute_predictions(mlp9, io3Test)
pred10 <- compute_predictions(mlp10, io3Test)
pred11 <- compute_predictions(mlp11, io4Test)
pred12 <- compute_predictions(mlp12, io4Test)



################ De-normalising #################


# Function to denormalize predictions
denormalize <- function(x, data) {
  max_val <- max(data$actual[1:400])
  min_val <- min(data$actual[1:400])
  return( x * (max_val - min_val) + min_val )
}

# Denormalize predictions for each model
denorm1 <- denormalize(pred1, io1)
head(pred1)
head(denorm1)

denorm2 <- denormalize(pred2, io2)
denorm3 <- denormalize(pred3, io3)
denorm4 <- denormalize(pred4, io4)
denorm5 <- denormalize(pred5, io1)
denorm6 <- denormalize(pred6, io1)
denorm7 <- denormalize(pred7, io2)
denorm8 <- denormalize(pred8, io2)
denorm9 <- denormalize(pred9, io3)
denorm10 <- denormalize(pred10, io3)
denorm11 <- denormalize(pred11, io4)
denorm12 <- denormalize(pred12, io4)

denorm_test_1 <- denormalize(io1Test, io1)
denorm_test_2 <- denormalize(io2Test, io2)
denorm_test_3 <- denormalize(io3Test, io3)
denorm_test_4 <- denormalize(io4Test, io4)


############# Evaluation matrices #################


library(MLmetrics)

# Define a function to calculate evaluation metrics
eval <- function(predictions, actual_values) {
  RMSE <- RMSE(predictions, actual_values)
  MAE <- MAE(predictions, actual_values)
  MAPE <- MAPE(predictions, actual_values)
  sMAPE <- mean(2 * abs(actual_values - predictions) / (abs(actual_values) + abs(predictions)))
  
  return(c(RMSE = RMSE, MAE = MAE, MAPE = MAPE, sMAPE = sMAPE))
}

# Compute evaluation metrics for each model
eval1 <- eval(denorm1, denorm_test_1$actual)
eval1
eval2 <- eval(denorm2, denorm_test_2$actual)
eval3 <- eval(denorm3, denorm_test_3$actual)
eval4 <- eval(denorm4, denorm_test_4$actual)
eval5 <- eval(denorm5, denorm_test_1$actual)
eval6 <- eval(denorm6, denorm_test_1$actual)
eval7 <- eval(denorm7, denorm_test_2$actual)
eval8 <- eval(denorm8, denorm_test_2$actual)
eval9 <- eval(denorm9, denorm_test_3$actual)
eval10 <- eval(denorm10, denorm_test_3$actual)
eval11 <- eval(denorm11, denorm_test_4$actual)
eval12 <- eval(denorm12, denorm_test_4$actual)

############ Comparison table ###################

library(dplyr)

# Define the evaluation metrics for each model
evaluation_metrics <- data.frame(
  Model = c("MLP1", "MLP2", "MLP3", "MLP4", "MLP5", "MLP6", "MLP7", "MLP8", "MLP9", "MLP10", "MLP11", "MLP12"),
  RMSE = c(eval1["RMSE"], eval2["RMSE"], eval3["RMSE"], eval4["RMSE"],
           eval5["RMSE"], eval6["RMSE"], eval7["RMSE"], eval8["RMSE"],
           eval9["RMSE"], eval10["RMSE"], eval11["RMSE"], eval12["RMSE"]),
  MAE = c(eval1["MAE"], eval2["MAE"], eval3["MAE"], eval4["MAE"],
          eval5["MAE"], eval6["MAE"], eval7["MAE"], eval8["MAE"],
          eval9["MAE"], eval10["MAE"], eval11["MAE"], eval12["MAE"]),
  MAPE = c(eval1["MAPE"], eval2["MAPE"], eval3["MAPE"], eval4["MAPE"],
           eval5["MAPE"], eval6["MAPE"], eval7["MAPE"], eval8["MAPE"],
           eval9["MAPE"], eval10["MAPE"], eval11["MAPE"], eval12["MAPE"]),
  sMAPE = c(eval1["sMAPE"], eval2["sMAPE"], eval3["sMAPE"], eval4["sMAPE"],
            eval5["sMAPE"], eval6["sMAPE"], eval7["sMAPE"], eval8["sMAPE"],
            eval9["sMAPE"], eval10["sMAPE"], eval11["sMAPE"], eval12["sMAPE"]), 
  Description = c(
    "Input: All features with lagged vectors in the order of 1, 3, 4, 2. Hidden Layer: 1 layer with 6 nodes, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 2, 1, 4, 3. Hidden Layer: 1 layer with 8 nodes, Activation: Tanh, Linear Output: No",
    "Input: All features with lagged vectors in the order of 1, 2, 3, 4. Hidden Layer: 1 layer with 5 nodes, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 4, 3, 2, 1. Hidden Layer: 1 layer with 4 nodes, Activation: Tanh, Linear Output: No",
    "Input: All features with lagged vectors in the order of 1, 3, 4, 2. Hidden Layers: 2 layers with 6 and 5 nodes respectively, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 1, 3, 4, 2. Hidden Layers: 3 layers with 8, 6, and 10 nodes respectively, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 2, 1, 4, 3. Hidden Layers: 2 layers with 12 and 8 nodes respectively, Activation: Logistic, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 2, 1, 4, 3. Hidden Layers: 3 layers with 8, 5, and 4 nodes respectively, Activation: Logistic, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 1, 2, 3, 4. Hidden Layers: 2 layers with 9 and 8 nodes respectively, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 1, 2, 3, 4. Hidden Layers: 3 layers with 6, 4, and 10 nodes respectively, Activation: Tanh, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 4, 3, 2, 1. Hidden Layers: 2 layers with 8 and 12 nodes respectively, Activation: Logistic, Linear Output: Yes",
    "Input: All features with lagged vectors in the order of 4, 3, 2, 1. Hidden Layers: 3 layers with 2, 5, and 3 nodes respectively, Activation: Tanh, Linear Output: Yes"
  ))



# Print the comparison table
print(evaluation_metrics)

min(evaluation_metrics$RMSE) 
min(evaluation_metrics$MAE)
min(evaluation_metrics$MAPE)
min(evaluation_metrics$sMAPE)

# best nns are 2 and 8 and 11 after inspecting through the evaluation metrics
#mlp2 is the most preferred

# Calculate evaluation metrics for MLP2
eval2 <- eval(denorm2, denorm_test_2$actual)
eval2


############## PLOTs ##############


library(ggplot2)

# PLOT USING G PLOT FOR PREDICTION OUTPUT VS DESIRED OUTPUT
plot_data <- data.frame(Actual = denorm_test_2$actual, Predicted = denorm2)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual Output", y = "Predicted Output", title = "Prediction vs. Actual Output")


par(mfrow=c(2, 1)) 
predts <- ts(denorm2)
plot(predts)
act <-ts(denorm_test_2$actual)
plot(act)



library(ggplot2)

# Create a dataframe for time series data
time_series_data <- data.frame(
  Date = seq(as.Date("2011-10-01"), by = "day", length.out = nrow(denorm_test_2)),
  Actual = denorm_test_2$actual,
  Predicted = denorm2
)


# Plotting the time series
ggplot(time_series_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(x = "Date", y = "Exchange Rate", title = "Actual vs. Predicted Exchange Rates") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()





