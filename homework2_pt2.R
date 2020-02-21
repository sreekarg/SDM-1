rm(list = ls())



library(DAAG)                               # once downloaded the library use this command to load the library
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
library(leaps)
library(tidyverse)





train <- read.delim("ticdata2000.txt", header = FALSE)
X_train <- as.matrix(train[,-c(86)])
Y_train <- train$V86
Y_train_hat <- as.matrix(Y_train)


X_test <- read.delim("ticeval2000.txt", header = FALSE)
Y_test <- read.delim("tictgts2000.txt", header = FALSE)
test <- X_test
Y_test_hat <- as.matrix(Y_test)

test$V86 <- Y_test 

########## Linear model###############
lm_fit <- lm(V86 ~., data= train)
summary(lm_fit)

lin_pred <- predict(lm_fit, X_test)
lin_pred_classify <- ifelse(lin_pred>0.5, 1.0, 0.0)
lin_err = mean((Y_test_hat - lin_pred_classify)^2)



###########  Forward selection ###############

regfit_fwd <- regsubsets(V86~., data = train, nbest = 1, nvmax = 85, method = "forward")
fwd_sum <- summary(regfit_fwd)

par(mfrow = c(2,2))
plot(fwd_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
plot(fwd_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")

plot(fwd_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(fwd_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "l")

fwd= which(fwd_sum$cp == min(fwd_sum$cp))
fwd_bic = which(fwd_sum$bic == min(fwd_sum$bic))
fwd_rss = which(fwd_sum$rss == min(fwd_sum$rss))


################  test error ######################
col_name = colnames(train)
fwd_coef = coef(regfit_fwd,fwd)


fwd_mod_pred = as.matrix(test[, col_name %in% names(fwd_coef)]) %*% fwd_coef[names(fwd_coef) %in% col_name]

fwd_pred_classify <- ifelse(fwd_mod_pred>0.5, 1.0, 0.0)
fwd_err = mean((Y_test_hat - fwd_pred_classify)^2)


###########  Backward selection ###############

regfit_bck <- regsubsets(V86~., data = train, nbest = 1, nvmax = 85, method = "backward")
bck_sum <- summary(regfit_bck)

par(mfrow = c(2,2))
plot(bck_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
plot(bck_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")

plot(bck_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(bck_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "l")

bck = which(bck_sum$cp == min(bck_sum$cp))
bck_bic = which(bck_sum$bic == min(bck_sum$bic))
bck_rss = which(bck_sum$rss == min(bck_sum$rss))


################  test error ######################
bck_coef = coef(regfit_bck,bck)


bck_mod_pred = as.matrix(test[, col_name %in% names(bck_coef)]) %*% bck_coef[names(bck_coef) %in% col_name]

bck_pred_classify <- ifelse(bck_mod_pred>0.5, 1.0, 0.0)
bck_err = mean((Y_test_hat - bck_pred_classify)^2)



################ Ridge  Regression  ################
ridge_mod <- cv.glmnet(X_train, Y_train_hat, alpha=0)
summary(ridge_mod)
plot(ridge_mod)
ridge_pred <- predict(ridge_mod, s = min(ridge_mod$lambda), newx = as.matrix(X_test), type = "response")

ridge_pred_classify <- ifelse(ridge_pred>0.5, 1.0, 0.0)
ridge_err = mean((Y_test_hat - ridge_pred_classify)^2)

################ Lasso  Regression  ################
lasso_mod <- cv.glmnet(X_train, Y_train_hat, alpha=1)
summary(lasso_mod)
plot(lasso_mod)
lasso_pred <- predict(lasso_mod, s = min(lasso_mod$lambda), newx = as.matrix(X_test), type = "response")

lasso_pred_classify <- ifelse(lasso_pred>0.5, 1.0, 0.0)
lasso_err = mean((Y_test_hat - lasso_pred_classify)^2)


method_error_value <- c(lin_err, ridge_err, lasso_err, bck_err, fwd_err)
names(method_error_value) <- c("linear","ridge","lasso","Backward","Forward")
barplot(method_error_value, xlab = 'error values for different approaches')

best_method = which(method_error_value == min(method_error_value))
method_error_value[best_method]




ridge_true = which(ridge_pred_classify == 1 & ridge_pred_classify == Y_test_hat)
ridge_res = which(Y_test_hat == 1)


lasso_true = which(lasso_pred_classify == 1)
lasso_res = which(Y_train_hat == 1)


method_error_value <- c(lin_err, ridge_err, lasso_err, bck_err, fwd_err)
names(method_error_value) <- c("linear","ridge","lasso","Backward","Forward")



print(" the accuracy of the linear regression model predicting those customers who are actually willing to buy is ")
lin_accuracy = length(which(lin_pred_classify == 1 & lin_pred_classify == Y_test_hat))/length(which(Y_test_hat == 1))
lin_accuracy



print(" the accuracy of the ridge regression model predicting those customers who are actually willing to buy is ")
ridge_accuracy = length(which(ridge_pred_classify == 1 & ridge_pred_classify == Y_test_hat )) /length(which(Y_test_hat == 1))
ridge_accuracy




print(" the accuracy of the lasso regression model predicting those customers who are actually willing to buy is ")
lasso_accuracy = length(which(lasso_pred_classify == 1 & lasso_pred_classify == Y_test_hat )) /length(which(Y_test_hat == 1))
lasso_accuracy



print(" the accuracy of the backward subset selection method predicting those customers who are actually willing to buy is ")



bck_accuracy = length(which(bck_pred_classify == 1 & bck_pred_classify == Y_test_hat )) /length(which(Y_test_hat == 1))
bck_accuracy


print(" the accuracy of the forward subset selection method predicting those customers who are actually willing to buy is ")
fwd_accuracy = length(which(fwd_pred_classify == 1 & fwd_pred_classify == Y_test_hat )) /length(which(Y_test_hat == 1))
fwd_accuracy


