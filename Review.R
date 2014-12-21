library(dplyr)
library(caret)

set.seed(98765)

if (!exists("dataTrain.raw")){dataTrain.raw <- read.csv('data/pml-training.csv')}
if (!exists("dataTest.raw")){dataTest.raw <- read.csv('data/pml-testing.csv')}

# Clean up the column set
pml_data_column_cleanup = function(df){
  # Drop purely admin related columns
  df <- df %>% select(-starts_with("X"))
  df <- df %>% select(-starts_with("raw_timestamp_"))
  df <- df %>% select(-starts_with("cvtd_timestamp"))
  df <- df %>% select(-starts_with("new_window"))
  df <- df %>% select(-starts_with("num_window"))
  
  # Drop columns that relate only to "New Window = Yes"
  df <- df %>% select(-starts_with("kurtosis_roll_"))
  df <- df %>% select(-starts_with("kurtosis_picth_"))
  df <- df %>% select(-starts_with("kurtosis_yaw_"))
  
  df <- df %>% select(-starts_with("skewness_roll_"))
  df <- df %>% select(-starts_with("skewness_pitch_"))
  df <- df %>% select(-starts_with("skewness_yaw_"))
  
  df <- df %>% select(-starts_with("max_roll_"))
  df <- df %>% select(-starts_with("max_picth_"))
  df <- df %>% select(-starts_with("max_yaw_"))
  
  df <- df %>% select(-starts_with("min_roll_"))
  df <- df %>% select(-starts_with("min_pitch_"))
  df <- df %>% select(-starts_with("min_yaw_"))
  
  df <- df %>% select(-starts_with("amplitude_roll_"))
  df <- df %>% select(-starts_with("amplitude_pitch_"))
  df <- df %>% select(-starts_with("amplitude_yaw_"))
  
  df <- df %>% select(-starts_with("var_total_accel_"))
  
  df <- df %>% select(-starts_with("avg_roll_"))
  df <- df %>% select(-starts_with("stddev_roll_"))
  df <- df %>% select(-starts_with("var_roll_"))
  
  df <- df %>% select(-starts_with("avg_pitch_"))
  df <- df %>% select(-starts_with("stddev_pitch_"))
  df <- df %>% select(-starts_with("var_pitch_"))
  
  df <- df %>% select(-starts_with("avg_yaw_"))
  df <- df %>% select(-starts_with("stddev_yaw_"))
  df <- df %>% select(-starts_with("var_yaw_"))
  
  df <- df %>% select(-starts_with("var_accel_"))
  
  df
}

dataTrain <- pml_data_column_cleanup(dataTrain.raw)
dataTest <- pml_data_column_cleanup(dataTest.raw)

if (!exists("dataTrain.model")){
  # Partition our data 75% training and 25% testing
  dataTrain.partition <- createDataPartition(y=dataTrain$classe, p=0.75, list=FALSE)
  dataTrain.train <- dataTrain[dataTrain.partition,]
  dataTrain.test  <- dataTrain[-dataTrain.partition,]
  
  # Implemented a Random Forest (limited to depth 100) training strategy
  dataTrain.model <- train(classe~., data=dataTrain.train, method="rf", ntree=100)
}

# Train Test prediction
dataTrain.test.predict <- predict(dataTrain.model, newdata=dataTrain.test)

# Test prediction (Validation)
dataTest.predict <- predict(dataTrain.model, newdata=dataTest)

# print(dataTrain.test.predict)
print(dataTest.predict)

# Calculate outside error rate and accuracy of the validation
outsideErrorRate.accuracy <- sum(dataTrain.test.predict == dataTrain.test$classe)/length(dataTrain.test.predict)
outsideErrorRate.error <- (1 - outsideErrorRate.accuracy)

print(
  ggplot(dataTrain.model) + ggtitle("Accuracy vs. Predictor")
  )

print(dataTrain.model)


# Wrap up with the answers for the submission
pml_write_files = function(x){
  dir.create(file.path(getwd(), 'submission'), showWarnings = FALSE)
  n = length(x)
  for(i in 1:n){
    filename = paste0("submission/problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}

pml_write_files(as.character(dataTest.predict))