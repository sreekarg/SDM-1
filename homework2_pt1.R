rm(list = ls())



library(DAAG)                 # once downloaded the library use this command to load the library
library(lattice)
library(MASS)
library("geneplotter")
library(ggplot2)
library(readr)
library("Hmisc")
library(corrplot)
library("dlookr")
library(dplyr)
library(tidyr)
library(ruler)
library(data.table)
library("ElemStatLearn")
library("class")
library("ISLR")
library(glmnet)
library(pls)

college_dataset <- College
college_dataset$Private = as.numeric(college_dataset$Private)
smp_siz = floor(0.75*nrow(college_dataset))

set.seed(123456)
train_ind = sample(seq_len(nrow(college_dataset)),size = smp_siz)
train <- na.fail(college_dataset[train_ind,])
test <- na.fail(college_dataset[-train_ind,])

X_train <- as.matrix(train[,-c(2)])
Y_train_hat <- train$Apps
X_test <- as.matrix(test[,-c(2)])
Y_test_hat <- test[,2]


############## Linear Model ##################### 

fit <- lm(Apps~ . , data = train)
summary(fit)
# trimmed_model <- lm(Apps ~Private+Accept+Top10perc+Top25perc+F.Undergrad+Room.Board+Expend+Grad.Rate, data = train)
# summary(trimmed_model)

##########   test predictions  ##########

lin_test_predict <- predict(fit, test)
head(lin_test_predict)

######### Test Error #########
lm_diff <- (lin_test_predict-Y_test_hat)^2
lin_test_error <- mean(lm_diff)

print("The test error for the linear model is ")
lin_test_error


###############################################
########## ridge regression  ##################

# ridge_mod <- glmnet(X_train, Y_train_hat, alpha=0)
# summary(ridge_mod)
# 
# head(ridge_mod$lambda)
# min(ridge_mod$lambda)
# 
# names(ridge_mod)
# head(coef(ridge_mod))
# dim(coef(ridge_mod))

# 
# predict(ridge_mod, s = min(ridge_mod$lambda), type = "coefficient")
# 
# ridge_mod$lambda[100]
# coef(ridge_mod)[,100]



?cv.glmnet

cv.out_ridge <- cv.glmnet(X_train, Y_train_hat, alpha = 0)


plot(cv.out_ridge)

names(cv.out_ridge)
bestlam_ridge <- cv.out_ridge$lambda.min
print("The best lambda chosen by cross validation is  ")
bestlam_ridge

ridge_pred <- predict(cv.out_ridge, s= bestlam_ridge, type = "coefficients")

ridge_pred_test <- predict(cv.out_ridge, s= bestlam_ridge, newx=X_test   ,type = "response")

ridge_diff = ridge_pred_test - Y_test_hat
ridge_test_err = mean((ridge_diff)^2)
print("The test error for the Ridge Regression model is ")
ridge_test_err


################################################
###################  The LASSO #################
# lasso_mod <- glmnet(X_train, Y_train_hat, alpha = 1)
# plot(lasso_mod)

cv.out_lasso <- cv.glmnet(X_train, Y_train_hat, alpha = 1)
plot(cv.out_lasso)

names(cv.out_lasso)
bestlam_lasso <- cv.out_lasso$lambda.min

print("The best lambda chosen by cross validation is  ")
bestlam_lasso

lasso_pred <- predict(cv.out_lasso, s= bestlam_lasso, type = "coefficients")

lasso_pred_test <- predict(cv.out_lasso, s= bestlam_lasso, newx=X_test   ,type = "response")

lasso_diff = lasso_pred_test - Y_test_hat
lasso_test_err = mean((lasso_diff)^2)

print("The test error for the Lasso model is ")
lasso_test_err
print("Number of non-zero coeffecient estimates is  ")
length(nonzeroCoef(lasso_pred)) - 1

########################################
############## PCR Model ###############


pcr_fit = pcr(Apps ~., data = train, scale = TRUE, validation = "CV")
pcr_sum <- summary(pcr_fit)

pcr_train_error_store <- c()
pcr_test_error_store <- c()
for (i in 1:17){
  pcr_pred_train = predict(pcr_fit, train, ncomp = i)
  pcr_pred_test = predict(pcr_fit, test, ncomp = i)
  pcr_train_error <- mean((pcr_pred_train - Y_train_hat)^2)
  pcr_test_error <- mean((pcr_pred_test- Y_test_hat)^2)
  pcr_train_error_store <- c(pcr_train_error_store, pcr_train_error)
  pcr_test_error_store <- c(pcr_test_error_store, pcr_test_error)
}

plot(pcr_train_error_store)

plot(pcr_test_error_store)
best_pcr_ncomp = which(pcr_test_error_store == min(pcr_test_error_store))

print('The test error for the PCR model is ')
pcr_test_error_store[best_pcr_ncomp]


print(' and k =  ') 
best_pcr_ncomp


#########################################
## Partial Least Squares (PLS)
#########################################
pls_fit = plsr(Apps ~., data = train, scale = TRUE, validation = "CV")
summary(pls_fit)
# validationplot(pls_fit, val.type = "MSEP")


pls_train_error_store <- c()
pls_test_error_store <- c()
for (i in 1:17){
  pls_pred_train = predict(pls_fit, train, ncomp = i)
  pls_pred_test = predict(pls_fit, test, ncomp = i)
  pls_train_error <- mean((pls_pred_train - Y_train_hat)^2)
  pls_test_error <- mean((pls_pred_test- Y_test_hat)^2)
  pls_train_error_store <- c(pls_train_error_store, pls_train_error)
  pls_test_error_store <- c(pls_test_error_store, pls_test_error)
}

plot(pls_train_error_store , type= 'line')

plot(pls_test_error_store)
best_pls_ncomp = which(pls_test_error_store == min(pls_test_error_store))

print('The test error for the PLS model is ')
pls_test_error_store[best_pls_ncomp]


print(' and  k =  ') 
best_pls_ncomp

method_error_value <- c(lin_test_error,ridge_test_err,lasso_test_err,pcr_test_error_store[best_pcr_ncomp],pls_test_error_store[best_pls_ncomp])
names(method_error_value) <- c("linear","ridge","lasso","pcr","pls")
barplot(method_error_value, xlab = 'error values for different approaches')

best_method = which(method_error_value == min(method_error_value))
method_error_value[best_method]

