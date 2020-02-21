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


################################################################
######################    pt3          #########################
################################################################

set.seed(100)

X_data = matrix(rnorm(1000*20),1000,20)
beta <- rnorm(20)

set.seed(20)
bet_ind = sample(1:20, 8)

beta[bet_ind] = 0
epsilon <- rnorm(20)

Y <- X_data%*%beta + epsilon


data <- data.frame(Y,X_data)

set.seed(900)

train_ind = sample(1:nrow(data), 100)

train <- data[train_ind,]
test <- data[-train_ind,]
names(test)


########## Best subset selection ##########
regfit_full <- regsubsets(Y~., data = train, nbest = 1, nvmax = 20, method = "exhaustive")
exh_sum = summary(regfit_full)

plot(exh_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
plot(exh_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")

plot(exh_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(exh_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "l")

min_var = which(exh_sum$cp == min(exh_sum$cp))
exh_bic = which(exh_sum$bic == min(exh_sum$bic))
exh_rss = which(exh_sum$rss == min(exh_sum$rss))
coef_betapred = data.frame(coef(regfit_full,min_var))
beta_pred_names = names(coef(regfit_full,min_var))


train_err = rep(NA, 20)
col_name = colnames(data)

for (i in 1:20) {
  coefi = coef(regfit_full, id = i)
  pred = as.matrix(train[, col_name %in% names(coefi)]) %*% coefi[names(coefi) %in% col_name]
  train_err[i] = mean((train$Y - pred)^2)
}


plot(train_err, type = "line", xlab = ' number of variables')

test_err = rep(NA, 20)
for (i in 1:20) {
  test_coefi = coef(regfit_full, id = i)
  test_pred = as.matrix(test[, col_name %in% names(test_coefi)]) %*% test_coefi[names(test_coefi) %in% col_name]
  test_err[i] = mean((test$Y - test_pred)^2)
}

plot(test_err, type = "l" , xlab = 'number of predictors')




