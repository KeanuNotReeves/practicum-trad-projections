library(readxl)
library(randomForest)
library(caret)

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

Admits$State <- NULL #random forests can't handle factors with more than 53 levels


summary(Admits)


#split the data into training & testing
ind <- sample(2, nrow(Admits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- Admits[ind == 1,]
testAdmits <- Admits[ind==2,]

#build a random forest.
set.seed(6)
rf <- randomForest(Enroll ~ ., data=trainAdmits, importance=TRUE, ntree=2000)
varImpPlot(rf)
rf$confusion
min(rf$err.rate)
PredRF <- predict(rf, testAdmits, type = "class")
PredRF
confusionMatrix(table(test15FA$Enroll, PredRF), positive = "Yes")

#conditional inference tree
set.seed(7)
CItree <- ctree(Enroll ~., data = trainAdmits)
CItree
predCI <- predict(CItree, testAdmits)
confusionMatrix(table(test15FA$Enroll, predCI), positive = "Yes")

#split the data into training & testing with Principle Components only
Prin <- sample(2, nrow(PrimComps), replace = TRUE, prob = c(0.8,0.2))
trainPrin <- PrimComps[Prin == 1,]
testPrin <- PrimComps[Prin==2,]
trainPrin$State <- NULL
testPrin$State <- NULL

#build a random forest on Principle components
set.seed(8)
rf1 <- randomForest(Enroll ~ ., data=trainPrin, importance=TRUE, ntree=2000)
varImpPlot(rf1)
rf1$confusion
min(rf1$err.rate)
PredRF1 <- predict(rf1, testPrin, type = "class")
confusionMatrix(table(testPrin$Enroll, PredRF1), positive = "Yes")

#conditional inference tree on principle components.
set.seed(9)
CItree1 <- ctree(Enroll ~., data = trainPrin)
CItree1
predCI1 <- predict(CItree1, testPrin)
confusionMatrix(table(testPrin$Enroll, predCI1), positive = "Yes")
