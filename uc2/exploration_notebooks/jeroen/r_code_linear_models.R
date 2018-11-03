library(MASS)
library(dplyr)


mydata <- read.csv("out.csv",sep = ";")

mydata <- mydata %>% select(-c('X'))

X <- mydata %>% select(-c('P', 'E', 'R', 'M', 'A'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]
 
test <- X[train_size:dim(X)[1],]

fit1 <- lm(PERMA ~ ., data=train)
fit2 <- lm(PERMA ~ 1, data=train)

PERMA_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)

p_PERMA_up <- predict(PERMA_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$PERMA,p_PERMA_up)


#------------------------------------------------------------------

X <- mydata %>% select(-c('PERMA', 'E', 'R', 'M', 'A'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]

test <- X[train_size:dim(X)[1],]

fit1 <- lm(P ~ ., data=train)
fit2 <- lm(P ~ 1, data=train)

P_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)


p_P_up <- predict(P_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$P,p_P_up)


#------------------------------------------------------------------

X <- mydata %>% select(-c('P', 'PERMA', 'R', 'M', 'A'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]

test <- X[train_size:dim(X)[1],]

fit1 <- lm(E ~ ., data=train)
fit2 <- lm(E ~ 1, data=train)

E_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)

p_E_up <- predict(E_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$E,p_E_up)


#------------------------------------------------------------------

X <- mydata %>% select(-c('P', 'E', 'PERMA', 'M', 'A'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]

test <- X[train_size:dim(X)[1],]

fit1 <- lm(R ~ ., data=train)
fit2 <- lm(R ~ 1, data=train)

R_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)

p_R_up <- predict(R_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$R,p_R_up)


#------------------------------------------------------------------


X <- mydata %>% select(-c('P', 'E', 'R', 'PERMA', 'A'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]

test <- X[train_size:dim(X)[1],]

fit1 <- lm(M ~ ., data=train)
fit2 <- lm(M ~ 1, data=train)

M_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)

p_M_up <- predict(M_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$M,p_M_up)


#------------------------------------------------------------------


X <- mydata %>% select(-c('P', 'E', 'R', 'M', 'PERMA'))

train_size <- floor(dim(X)[1]*0.8)

train <- X[0:train_size,]

test <- X[train_size:dim(X)[1],]

fit1 <- lm(A ~ ., data=train)
fit2 <- lm(A ~ 1, data=train)

A_up <- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2),trace=0)

p_A_up <- predict(A_up,test)

rsq <- function (x, y) cor(x, y) ^ 2

rsq(test$A,p_A_up)

#------------------------------------------------------------------

test <- mydata[train_size:dim(X)[1],]

#summary(PERMA_up)
rsq(test$PERMA,p_PERMA_up)

#summary(P_up)
rsq(test$P,p_P_up)

#summary(E_up)
rsq(test$E,p_E_up)

#summary(R_up)
rsq(test$R,p_R_up)

#summary(M_up)
rsq(test$M,p_M_up)

#summary(A_up)
rsq(test$A,p_A_up)


library(ggplot2)

data1 <- data.frame(true_PERMA=test$PERMA,predicted_PERMA=p_PERMA_up)[-c(18),]

ggplot(data1, aes(x=true_PERMA, y=predicted_PERMA)) +
  ggtitle("PERMA Score True Value vs Predicted Value") +
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")


data2 <- data.frame(true_R=test$R,predicted_R=p_R_up)

ggplot(data2, aes(x=true_R, y=predicted_R)) +
  ggtitle("(R)elationship Score True Value vs Predicted Value") +
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue") 




