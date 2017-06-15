library(readxl)
library(randomForest)
library(caret)
library(adabag)
library(e1071)
library(party)
library(neuralnet)
library(h2o)
library(dummies)
library(dummy)
library(party)

#bring in the 15FA data
X15FA <- as.data.frame(read_excel("D:/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))

#set variable types
X15FA$Gender <- as.factor(X15FA$Gender)
X15FA$Ethnic <- as.factor(X15FA$Ethnic)
X15FA$Religion <- as.factor(X15FA$Religion)
X15FA$Career_Goals <- as.factor(X15FA$Career_Goals)
X15FA$FA_Intent <- as.factor(X15FA$FA_Intent)
X15FA$First_Gen <- as.factor(X15FA$First_Gen)
X15FA$HS_Type <- as.factor(X15FA$HS_Type)
X15FA$Visit <- as.factor(X15FA$Visit)
X15FA$Rating <- as.factor(X15FA$Rating)
X15FA$Housing_Desired <- as.factor(X15FA$Housing_Desired)
X15FA$Legacy <- as.factor(X15FA$Legacy)
X15FA$Enroll <- as.factor(X15FA$Enroll)
X15FA$Regis_Position <- as.factor(X15FA$Regis_Position)


#perform EDA on the dataset
str(X15FA)
summary(X15FA)
hist(X15FA$Composite_Score, main = "Composite Score Distribution", xlab = "Composite Score", ylab = "Number of Students")
hist(X15FA$Distance, main = "Distance From Campus Distribution", xlab = "Distance (miles)", ylab = "Number of Students")

#Clean the data, removing or replacing NA's
which((is.na(X15FA$GPA)))
X15FA$GPA[c(428,1666,2067,2384,2508,3135)] <- median(X15FA$GPA, na.rm = TRUE)
which((is.na(X15FA$GPA)))

which(is.na(X15FA$Distance))
X15FA$Distance[c(6,30,200,224,252,293,416,428,480,598,742,749,918,949,1015,1076,1231,1321,1466,1666,1693,1730,1936,2002,2014,2018,2067,2105,2112,2182,2249,2382,2415,2508,2618,2850,2851,2927,3184,3186,3222,3421)] <- max(X15FA$Distance, na.rm = TRUE)
which(is.na(X15FA$Distance))

which(is.na(X15FA$Career_Goals))
X15FA$Career_Goals[c(556,851,1092,1310,1874,2013,2180,2980,3331,3335)] <- "UND"
which(is.na(X15FA$Career_Goals))

which(is.na(X15FA$State))
X15FA$State[c(6,224,252,1076,1231,1466,1620,1666,1693,1730,1802,1936,2002,2014,2018,2067,2105,2249,2916)] <- "International"
which(is.na(X15FA$State))
X15FA$State <- as.factor(X15FA$State)

#remove the columns with majority missing values
X15FA$`HS_Rank_%` <- NULL
X15FA$Student_AGI <- NULL
X15FA$EFC <- NULL

summary(X15FA)


#split the data into training & testing
ind <- sample(2, nrow(X15FA), replace = TRUE, prob = c(0.8,0.2))
train15FA <- X15FA[ind == 1,]
test15FA <- X15FA[ind==2,]


#bagging & boosting.
#bagging method
set.seed(1)
train15FA$Enroll <- as.factor(train15FA$Enroll)
test15FA$Enroll <- as.factor(test15FA$Enroll)
X15FA.bagg <- bagging(Enroll ~ ., data = train15FA, mfinal = 10)
X15FA.bagg$importance

#test
set.seed(2)
X15FA.predbagging <- predict.bagging(X15FA.bagg, newdata = test15FA)
X15FA.predbagging$confusion
X15FA.predbagging$error

#10-fold cross validation
set.seed(3)
X15FA.baggingcv <- bagging.cv(Enroll ~., v=10, data = X15FA, mfinal = 100)
X15FA.baggingcv$confusion
X15FA.baggingcv$error

#boosting method
set.seed(4)
X15FA.boost <- boosting(Enroll ~., data = train15FA, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
X15FA.predboost <- predict.boosting(X15FA.boost, newdata = test15FA)
X15FA.predboost$confusion
X15FA.predboost$error

#boosting with cross validation
set.seed(5)
X15FA.boostcv <- boosting.cv(Enroll~., v=10, data = X15FA, mfinal=100)
X15FA.boostcv$confusion
X15FA.boostcv$error