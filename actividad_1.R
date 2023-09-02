
################################################################################
############################## Cargar librerías ################################
################################################################################

install.packages('corrplot')
install.packages('fastDummies')
install.packages('caret')
install.packages('glmnet')

library(corrplot)
library(fastDummies)
library(caret)
library(dplyr)
library(tidyr)
library(glmnet)

# Fijar semilla para la reproducibilidad de resultados
set.seed(1)


################################################################################
################################ Leer los datos ################################
################################################################################

# Leer el archivo CSV; establecer celdas faltanes como NAs
df <- read.csv("Real_Estate_Sales_2001-2020_GL.csv", 
               na.strings = c("", "NA", "N/A", "missing"))


################################################################################
############################### Análisis inicial ###############################
################################################################################

# Analizar características pricipales del dataframe
View(df)
head(df) # Primeras 5 filas del dataframe
colnames(df) # Columnas del dataframe
summary(df) # Características principales

# Número de NAs por cada columna
na_counts <- colSums(is.na(df))
na_summary <- data.frame(row.names = colnames(df), NA_Count = na_counts)
na_summary

# Tamaño del dataframe
nrow(df)

# Eliminar valores atípicos de Assessed.Value
z_scores <- abs(scale(df$Assessed.Value))
threshold <- 3
df <- df[z_scores <= threshold, ]
# Histograma de Assessed.Value
hist(df$Assessed.Value, breaks = 100, main = "Histograma en escala logarítmica")

# Eliminar valores atípicos de Sale.Amount
z_scores <- abs(scale(df$Sale.Amount))
threshold <- 3
df <- df[z_scores <= threshold, ]
# Histograma de Sale.Amount (variable a predecir)
hist(df$Sale.Amount, breaks = 100, main = "Histograma en escala logarítmica")


################################################################################
############################# Procesamiento inicial ############################
################################################################################

###################### 1) Eliminar columnas innecesarias #######################

columns_to_remove <- c("Date.Recorded", "Serial.Number", "Town", "Address", 
                       "Sales.Ratio", "Non.Use.Code", "Assessor.Remarks", 
                       "OPM.remarks", "Location")
df <- df[, !(colnames(df) %in% columns_to_remove)]
head(df)


############################### 2) Eliminar filas ############################## 

# Eliminar filas con NAs
df <- na.omit(df)

# Eliminar valores atípicos (ventas de inmuebles por valor de 0)
rows_to_remove <- df$Assessed.Value == 0 | df$Sale.Amount == 0
df <- df[!rows_to_remove, ]


################# 3) Transformar columnas categoricas a dummy ##################

# Para K clases de una variable categorica nos quedamos con K-1 variables dummy 
# para evitar la multicolinealidad; también se eliminan las variables originales
df <- dummy_cols(df, select_columns = c("Property.Type", "Residential.Type"),
                 remove_selected_columns = TRUE, remove_first_dummy  = TRUE)

# Visualizar el número total de columnas
colnames(df)


########################## 4) Convertir años a enteros #########################

# Lista de años unicos
unique_years <- sort(unique(df$List.Year))

# Crear mapping de años a numeros secuenciales enteros del 1 al 15
year_to_number <- setNames(seq_along(unique_years), unique_years)

# Convertir años a enteros
df$List.Year <- year_to_number[as.character(df$List.Year)]
head(df)

############## 5) Normalizar variables numéricas (no categóricas) ##############

# Almacenar media y desviación estandar originales de la variable a predecir
y_mean <- mean(df$Sale.Amount)
y_sd <- sd(df$Sale.Amount)

# Crear función de estandarización
standardize = function(x){
  z <- (x - mean(x)) / sd(x)
  return( z)
}

#  Aplicar función de estandarización a variables numérica (no dummy)
df[2:3] <- apply(df[2:3], 2, standardize)
head(df)


################################################################################
##################### 6) Análisis y visualización de datos #####################
################################################################################

# Analizar correlaciones
M = cor(df)
png("Correlaciones.png", width=1000, height=1000)
corrplot(M, method = 'number')
dev.off()


################################################################################
########################## 7) Procesado de los datos ###########################
################################################################################

# Mezclar los datos de forma aleatoria (necesario ya que existe un orden 
# temporal ascendiente en el precio de las viviendas)
df <- df[sample(1:nrow(df)),]

# Dividir datos en grupos train y test
train_index <- createDataPartition(df$Sale.Amount, p = 0.8, list = FALSE, 
                                   times = 1)

train_data <- df[train_index, ] # Puede dar un error pero no afecta al resultado
test_data <- df[-train_index, ]

# Contar el número de filas de cada grupo
nrow(train_data)
nrow(test_data)


# Eliminar valores atípicos fuera de -2 o 2 para el conjunto de Train
outlier_threshold <- 2
outliers <- abs(train_data$Sale.Amount) > outlier_threshold
train_data <- train_data[!outliers, ]
outliers <- abs(train_data$Assessed.Value) > outlier_threshold
train_data <- train_data[!outliers, ]
nrow(train_data) # 480328


################################################################################
########################### 8) Definión de  funciones ##########################
################################################################################

# Función que permite evaluar distintos modelos de regresión
perform_regression_and_save_plots <- function(train_data, test_data, 
                                              formula_string, 
                                              regression_function, folder_name)
  {
  # Crear nuevo directorio si no existe
  if (!dir.exists(folder_name)) {
    dir.create(folder_name)
  }
  
  # Realizar regresión
  lm_model <- regression_function(formula_string, data = train_data)
  
  # Imprimir coeficientes
  coefficients <- coef(lm_model)
  print("\nResumen del modelo")
  print(summary(lm_model))
  
  # Extraer residuos
  res <- lm_model$residuals
  
  # Plot Residuaos vs Ajustados
  png(file.path(folder_name, "Residuos_vs_Valores_Ajustados.png"))
  plot(fitted(lm_model), res, main="Residuos vs Valores Ajustados")
  abline(0, 0)
  dev.off()
  
  # Plot Residuos Estandarizados  vs Valores Ajustados
  png(file.path(folder_name, "Residuos_Estandarizados_vs_Valores_Ajustados.png"))
  plot(lm_model, which = 1, main="Residuos Estandarizados vs Valores Ajustados")
  dev.off()
  
  # Plot Q-Q
  png(file.path(folder_name, "QQ_Plot.png"))
  plot(lm_model, which = 2, main="Quantile-Quantile")
  dev.off()
  
  # Plot Escala-Localizacion
  png(file.path(folder_name, "Escala_Localizacion.png"))
  plot(lm_model, which = 3, main="Escala-Localización")
  dev.off()
  
  # Plot Palanca (Leverage)
  png(file.path(folder_name, "Palanca.png"))
  plot(lm_model, which = 5)
  dev.off()

  return(lm_model)
}

# Definir métricas de evaluación
eval_results <- function(actual_values, predictions, df) {
  
  # Reescalar a valores a la escala original
  predictions <- predictions * y_sd + y_mean
  actual_values <- actual_values * y_sd + y_mean
  
  # Calcular R2
  r2 <- cor(actual_values, predictions)^2
  
  # Calcular Mean Absolute Error (MAE)
  mae <- mean(abs(actual_values - predictions))
  
  # Calcular Mean Absolute Percentage Error (MAPE)
  mape <- mean(abs((actual_values - predictions)/actual_values))*100
  
  # Calcular Root Mean Squared Error (RMSE)
  rmse <- sqrt(mean((actual_values - predictions)^2))
  
  # Imprimir métricas de evaluación
  cat("R-squared:", r2, "\n")
  cat("Mean Absolute Error:", mae, "\n")
  cat("Mean Absolute Percentage Error:", mape, "\n")
  cat("Root Mean Squared Error:", rmse, "\n")
}


################################################################################
####################### Tests estadísticos y resultados ########################
################################################################################

############################# 1) Regresión lineal ##############################

# Construir el string de formula
formula_string = "Sale.Amount~Assessed.Value"
# Entrenar el modelo
lm_model <- perform_regression_and_save_plots(train_data, test_data, formula_string, 
                                              lm, "regresion_lineal")

# Predicción y evaluación de los datos de train
predictions_train <- predict(lm_model, newdata = train_data)
eval_results(train_data$Sale.Amount, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(lm_model, newdata = test_data)
eval_results(test_data$Sale.Amount, predictions_test, test_data)


##################### 2) Regresión lineal multidimensional #####################

# Extraer los nombres de columnas excepto por "Sale.Amount" (variable a predecir)
predictor_columns <- setdiff(colnames(df), "Sale.Amount")
# Construir el string de formula
formula_string <- paste("`Sale.Amount`~", paste("`", predictor_columns, "`", 
                                                collapse = "+", sep = ""), 
                        sep = "")
# Entrenar el modelo
lm_model <- perform_regression_and_save_plots(train_data, test_data, 
                                          formula_string, lm, 
                                          "regresion_lineal_multidimensional")

# Predicción y evaluación de los datos de train
predictions_train <- predict(lm_model, newdata = train_data)
eval_results(train_data$Sale.Amount, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(lm_model, newdata = test_data)
eval_results(test_data$Sale.Amount, predictions_test, test_data)


################# 3) Regresión polinómica multidimensional #####################

# Construir el string de formula para regresión polinómica de grado 2
formula_string <- paste("`Sale.Amount`~", paste("poly(`", predictor_columns, "`, 
                                                2, raw = TRUE)", collapse = "+", 
                                                sep = ""), sep = "")
# Entrenar el modelo
lm_model <- perform_regression_and_save_plots(train_data, test_data, formula_string, 
                                              lm, "regresion_polinomica")

# Predicción y evaluación de los datos de train
predictions_train <- predict(lm_model, newdata = train_data)
eval_results(train_data$Sale.Amount, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(lm_model, newdata = test_data)
eval_results(test_data$Sale.Amount, predictions_test, test_data)


########################### 4) Regresión ridge (L2) ############################

# Convertir DataFrame a matriz para usar la libreria glmnet
x_train <- unname(as.matrix(train_data[,-3])) # Convertir a una matriz
y_train <- train_data$Sale.Amount # Convertir a un vector
x_test <- unname(as.matrix(test_data[,-3]))
y_test <- test_data$Sale.Amount

# Obtener valor optimo de lambda
lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- ridge_reg$lambda.min
optimal_lambda # 0.001

# Entrenar el modelo ridge (alpha = 0) con la lambda optima
ridge_model <- glmnet(x_train, y_train, alpha = 0, family = 'gaussian', 
                      lambda = optimal_lambda)
summary(ridge_model)

# Predicción y evaluación de los datos de train
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x_train)
eval_results(y_train, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test_data)


########################### 5) Regresión Lasso (L1) ############################

# Obtener valor optimo de lambda
lambdas <- 10^seq(2, -3, by = -.1)
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, 
                       standardize = TRUE, nfolds = 5)
optimal_lambda <- lasso_reg$lambda.min
optimal_lambda # 0.001

# Entrenar el modelo lasso (alpha = 1) con la lambda optima
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = optimal_lambda, 
                      standardize = TRUE)
summary(lasso_reg)

# Predicción y evaluación de los datos de train
predictions_train <- predict(lasso_reg, s = optimal_lambda, newx = x_train)
eval_results(y_train, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(lasso_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test_data)


########################### 6) Elastic-Net (L1 + L2) ###########################

# Especificar cómo se llevará a cabo la validación cruzada repetida
train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Construir el modelo de Elastic-Net en el que se prueba un rango de posibles 
# valores alfa y lambda y se selecciona su valor óptimo
elastic_reg <- train(Sale.Amount ~ .,
                     data = train_data,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)
# Mejor parámetro
elastic_reg$bestTune

# Predicción y evaluación de los datos de train
predictions_train <- predict(elastic_reg, x_train)
eval_results(y_train, predictions_train, train_data)

# Predicción y evaluación de los datos de test
predictions_test <- predict(elastic_reg, x_test)
eval_results(y_test, predictions_test, test_data)


############################ 7) Regresión Logística ############################

# Se utiliza una variable categórica (Property.Type_Residential) para testear la
# regresión logística

# Entrenar el modelo
log_reg_model <- train(
  Property.Type_Residential ~ .,
  data = train_data,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = trainControl(method = "cv", number = 5)
)

# Predecir los datos de test y convertir a resultados binarios
binary_predictions <- as.integer(predict(log_reg_model, newdata = test_data) > 0.5)

# Evaluar el modelo y mostrar matriz de confusión
conf_matrix <- confusionMatrix(table(binary_predictions, test_data$Property.Type_Residential))
print(conf_matrix)

