library(readxl)
library(caret)
library(adabag)
library(e1071)
library(party)


#bring in the 15FA data
X15FA <- as.data.frame(read_excel("D:/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))
X16FA <- as.data.frame(read_excel("D:/Practicum/Project Data/16FA/16FA Final.xlsx",sheet = "Final"))

Admits <- rbind(X15FA, X16FA)

#set variable types
Admits$Gender <- as.factor(Admits$Gender)
Admits$Ethnic <- as.factor(Admits$Ethnic)
Admits$Religion <- as.factor(Admits$Religion)
Admits$FA_Intent <- as.factor(Admits$FA_Intent)
Admits$First_Gen <- as.factor(Admits$First_Gen)
Admits$HS_Type <- as.factor(Admits$HS_Type)
Admits$Visit <- as.factor(Admits$Visit)
Admits$Rating <- as.factor(Admits$Rating)
Admits$Legacy <- as.factor(Admits$Legacy)
Admits$Enroll <- as.factor(Admits$Enroll)
Admits$Regis_Position <- as.factor(Admits$Regis_Position)


#perform EDA on the dataset
str(Admits)
summary(Admits)
hist(Admits$Composite_Score, main = "Composite Score Distribution", xlab = "Composite Score", ylab = "Number of Students")
hist(Admits$Distance, main = "Distance From Campus Distribution", xlab = "Distance (miles)", ylab = "Number of Students")

#Clean the data, removing or replacing NA's
which((is.na(Admits$GPA)))
Admits$GPA[is.na(Admits$GPA)] <- with(Admits, median(Admits$GPA, na.rm = TRUE))
which((is.na(Admits$GPA)))

which(is.na(Admits$Distance))
Admits$Distance[is.na(Admits$Distance)] <- with(Admits, max(Admits$Distance, na.rm = TRUE))
which(is.na(Admits$Distance))

which(is.na(Admits$State))
Admits$State[is.na(Admits$State)] <- "International"
which(is.na(Admits$State))
Admits$State <- as.factor(Admits$State)

summary(Admits)

Admits <- SMOTE(Enroll ~ ., Admits, perc.over = 500)


#split the data into training & testing
ind3 <- sample(2, nrow(Admits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- Admits[ind3 == 1,]
testAdmits <- Admits[ind3 == 2,]


#bagging & boosting.
#bagging method
set.seed(1)
trainAdmits$Enroll <- as.factor(trainAdmits$Enroll)
testAdmits$Enroll <- as.factor(testAdmits$Enroll)
Admits.bagg <- bagging(Enroll ~ ., data = trainAdmits, mfinal = 10)
Admits.bagg$importance

#test
set.seed(2)
Admits.predbagging <- predict.bagging(Admits.bagg, newdata = testAdmits)
Admits.predbagging$confusion
Admits.predbagging$error

#10-fold cross validation
set.seed(3)
Admits.baggingcv <- bagging.cv(Enroll ~., v=10, data = Admits, mfinal = 100)
Admits.baggingcv$confusion
Admits.baggingcv$error

#boosting method
set.seed(4)
Admits.boost <- boosting(Enroll ~., data = trainAdmits, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Admits.predboost <- predict.boosting(Admits.boost, newdata = testAdmits)
Admits.predboost$confusion
Admits.predboost$error

#boosting with cross validation
set.seed(5)
Admits.boostcv <- boosting.cv(Enroll~., v=10, data = Admits, mfinal=100)
Admits.boostcv$confusion
Admits.boostcv$error
Admits$Prediction <- Admits.boostcv$class
View(Admits)
