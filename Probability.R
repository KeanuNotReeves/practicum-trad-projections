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


#build a linear regression model
#train15FA$Enroll <- as.factor(train15FA$Enroll)
#reg15FA <- lm(formula= Enroll ~ ., data = train15FA)
#predLR <- predict.lm(reg15FA, newdata = test15FA, interval = "predict")
#confusionMatrix(predLR, reference = X15FA$Enroll, positive = "1")

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


#build a random forest.
set.seed(6)
rf <- randomForest(as.factor(Enroll) ~ ., data=trainNum, importance=TRUE, ntree=2000)
varImpPlot(rf)
PredRF <- predict(rf, testNum, type = "class")
confusionMatrix(table(testNum$Enroll, PredRF))
min(rf$err.rate)

#conditional inference tree
set.seed(7)
CItree <- ctree(Enroll ~., data = train15FA)
CItree
predCI <- predict(CItree, test15FA)
confusionMatrix(table(test15FA$Enroll, predCI), positive = "Yes")


#SVM
set.seed(8)
SVM15FA <- svm(Enroll ~ ., data = train15FA, kernel = "radial", cost = 1, gamma = 1/ncol(train15FA))
summary(SVM15FA)
SVMpred <- predict(SVM15FA, test15FA[, !names(test15FA) %in% c("Enroll")])
svm.table <- table(SVMpred, test15FA$Enroll)
svm.table
classAgreement(svm.table)
confusionMatrix(svm.table, positive = "Yes")

plot(SVM15FA, train15FA, Enroll ~ ., slice = list(Enroll = 2))


#neural network with neuralnet package
#set.seed(8)
#train15FA$EnrollYes <- train15FA$Enroll == "Yes"
#train15FA$EnrollNo <- train15FA$Enroll == "No"
#network <- neuralnet(EnrollYes + EnrollNo ~ HSType + DistanceFromCampus + HSGPA + Visit + RegisPosition + FAFSASubmission, data = train15FA, hidden = 3)
#network$result.matrix
#plot(network)
#net.predict <- compute(network, test15FA[-17])$net.result
#net.prediction <- c("EnrollYes", "EnrollNo")[apply(net.predict, 1, which.max)]
#predict.table <- table(test15FA$Enroll, net.prediction)
#predict.table

#Neurl network with H2O package
#start h20 instance
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
pathtodata <- paste0(normalizePath("D:/Practicum/Project Data/15FA/"),"/15FA Final Data.xlsx", sheet = "Final 2")
write.table(x=X15FA, file = pathtodata, row.names = F, col.names = T)
dat_h2o <- h2o.importFile(path = pathtodata, destination_frame = "X15FA")
h2o.describe(dat_h2o)


#model1 with a deep learning network (three layers of 50 nodes)
set.seed(54321)
model1 <- h2o.deeplearning(x=2:18,
                           y=19,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model1

#model2 with three layers of 100 nodes
model2 <- h2o.deeplearning(x=2:18,
                           y=19,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(100,100,100),
                           epochs = 50,
                           nfolds = 10)
model2

#model3 with three layers of 50 nodes and a Tanh activation
model3 <- h2o.deeplearning(x=2:18,
                           y=19,
                           training_frame = dat_h2o,
                           activation = "TanhWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model3

#model4 with three layers of 50 nodes and a Tanh activation, no dropout
set.seed(14)
model4 <- h2o.deeplearning(x=2:18,
                           y=19,
                           training_frame = dat_h2o,
                           activation = "Tanh",
                           input_dropout_ratio = 0.2,
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model4

#model5 with three layers of 100 nodes and a Tanh activation, no dropout
set.seed(14)
model5 <- h2o.deeplearning(x=2:18,
                           y=19,
                           training_frame = dat_h2o,
                           activation = "Tanh",
                           input_dropout_ratio = 0.2,
                           hidden = c(100,100,100),
                           epochs = 50,
                           nfolds = 10)
model5



#Neurl network with H2O package
#start h20 instance
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
df <- h2o.importFile(paste0(path = normalizePath("D:/Practicum/Project Data/15FA/"),"/15FA Final Data.xlsx", sheet = "Final 2"))
dim(df)
df
splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%


#model6 with a deep learning network (three layers of 50 nodes)
model6 <- h2o.deeplearning(x=2:18,
                           y=19,
                           set.seed(789),
                           training_frame = train,
                           validation_frame = test,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10,
                           variable_importances = T)
model6
head(as.data.frame(h2o.varimp(model6)))

h2o.performance(model6, train = T)
h2o.performance(model6, valid = T)
h2o.performance(model6, newdata = train)
h2o.performance(model6, newdata = valid)
h2o.performance(model6, newdata = test)

DLpred <- h2o.predict(model6, test)
DLpred
test$Accuracy <- DLpred$predict == test$Enroll
1-mean(test$Accuracy)


#set variable types for PCA
NUM15FA <- X15FA
NUM15FA$Gender <- as.numeric(NUM15FA$Gender)
NUM15FA$Ethnic <- as.numeric(NUM15FA$Ethnic)
NUM15FA$Religion <- as.numeric(NUM15FA$Religion)
NUM15FA$Career_Goals <- as.numeric(NUM15FA$Career_Goals)
NUM15FA$FA_Intent <- as.numeric(NUM15FA$FA_Intent)
NUM15FA$First_Gen <- as.numeric(NUM15FA$First_Gen)
NUM15FA$HS_Type <- as.numeric(NUM15FA$HS_Type)
NUM15FA$Visit <- as.numeric(NUM15FA$Visit)
NUM15FA$Rating <- as.numeric(NUM15FA$Rating)
NUM15FA$Housing_Desired <- as.numeric(NUM15FA$Housing_Desired)
NUM15FA$Legacy <- as.numeric(NUM15FA$Legacy)
NUM15FA$Enroll <- as.numeric(NUM15FA$Enroll)
NUM15FA$Regis_Position <- as.numeric(NUM15FA$Regis_Position)
NUM15FA$State <- as.numeric(NUM15FA$State)

#split the data into training & testing based on numerical data frame
ind2 <- sample(2, nrow(NUM15FA), replace = TRUE, prob = c(0.8,0.2))
trainNum <- PrimComps[ind2 == 1,]
testNum <- PrimComps[ind2 == 2,]

#Principle Component Analysis
sub <-subset(NUM15FA,select = -c(ID, Enroll))
pca <- prcomp(sub, scale. = T)
pca$rotation
dim(pca$x)
biplot(pca, scale = 0)
std_dev <- pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)
plot(prop_varex, main = "Scree Plot", xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")
plot(cumsum(prop_varex), main = "Cumulative Sum of Variance", xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b")

PrimComps <- subset(X15FA, select = c(Rating, State, Visit, Time_between_App_and_Term, Career_Goals, Regis_Position, Enroll))
PrimComps$Rating <- as.factor(PrimComps$Rating)
PrimComps$State <- as.factor(PrimComps$State)
PrimComps$Visit <- as.factor(PrimComps$Visit)

#split the data into training & testing based on PCA set
ind3 <- sample(2, nrow(PrimComps), replace = TRUE, prob = c(0.8,0.2))
trainPrim <- PrimComps[ind3 == 1,]
testPrim <- PrimComps[ind3 == 2,]

#build a linear regression model
#train15FA$Enroll <- as.factor(train15FA$Enroll)
#reg15FA <- lm(formula= Enroll ~ ., data = train15FA)
#predLR <- predict.lm(reg15FA, newdata = test15FA, interval = "predict")
#confusionMatrix(predLR, reference = X15FA$Enroll, positive = "1")

#bagging & boosting.
#bagging method
set.seed(1)
trainPrim$Enroll <- as.factor(trainPrim$Enroll)
testPrim$Enroll <- as.factor(testPrim$Enroll)
Prim.bagg <- bagging(Enroll ~ ., data = trainPrim, mfinal = 10)
Prim.bagg$importance

#test
set.seed(2)
Prim.predbagging <- predict.bagging(Prim.bagg, newdata = testPrim)
Prim.predbagging$confusion
Prim.predbagging$error

#10-fold cross validation
set.seed(3)
Prim.baggingcv <- bagging.cv(Enroll ~., v=10, data = PrimComps, mfinal = 100)
Prim.baggingcv$confusion
Prim.baggingcv$error

#boosting method
set.seed(4)
Prim.boost <- boosting(Enroll ~., data = trainPrim, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Prim.predboost <- predict.boosting(Prim.boost, newdata = testPrim)
Prim.predboost$confusion
Prim.predboost$error

#boosting with cross validation
set.seed(5)
Prim.boostcv <- boosting.cv(Enroll~., v=10, data = PrimComps, mfinal=100)
Prim.boostcv$confusion
Prim.boostcv$error


#build a random forest.
set.seed(6)
Primrf <- randomForest(as.factor(Enroll) ~ ., data=trainPrim, importance=TRUE, ntree=2000)
varImpPlot(Primrf)
PrimPredRF <- predict(Primrf, testPrim, type = "class")
confusionMatrix(table(testPrim$Enroll, PrimPredRF))
min(Primrf$err.rate)

#conditional inference tree
set.seed(7)
PrimCItree <- ctree(Enroll ~., data = trainPrim)
PrimCItree
PrimpredCI <- predict(PrimCItree, testPrim)
confusionMatrix(table(testPrim$Enroll, PrimpredCI), positive = "Yes")


#SVM
set.seed(8)
SVMPrim <- svm(Enroll ~ ., data = trainPrim, kernel = "radial", cost = 1, gamma = 1/ncol(trainPrim))
summary(SVMPrim)
SVMpredPrim <- predict(SVMPrim, testPrim[, !names(testPrim) %in% c("Enroll")])
Prim.svm.table <- table(SVMpredPrim, testPrim$Enroll)
Prim.svm.table
classAgreement(Prim.svm.table)
confusionMatrix(Prim.svm.table, positive = "Yes")

