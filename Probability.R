library(readxl)
library(randomForest)
library(caret)
library(adabag)
library(e1071)
library(party)
library(neuralnet)
library(h2o)
library(dummies)
#bring in the 15FA data
X15FA <- as.data.frame(read_excel("D:/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))

#set variables
#X15FA$Gender <- as.factor(X15FA$Gender)
#X15FA$Ethnic <- as.factor(X15FA$Ethnic)
#X15FA$Religion <- as.factor(X15FA$Religion)
#X15FA$`Career Goals` <- as.factor(X15FA$`Career Goals`)
#X15FA$`FA Intent` <- as.factor(X15FA$`FA Intent`)
#X15FA$`First Gen` <- as.factor(X15FA$`First Gen`)
#X15FA$`HS Type` <- as.factor(X15FA$`HS Type`)
#X15FA$`Res Stat` <- as.factor(X15FA$`Res Stat`)
#X15FA$Visit <- as.factor(X15FA$Visit)
#X15FA$`Pre-Reg` <- as.factor(X15FA$`Pre-Reg`)
#X15FA$`Housing Desired` <- as.factor(X15FA$`Housing Desired`)
#X15FA$Legacy <- as.factor(X15FA$Legacy)
#X15FA$Enroll <- as.numeric(X15FA$Enroll)
#X15FA$`Regis Position` <- as.factor(X15FA$`Regis Position`)


#perform EDA on the dataset
str(X15FA)
summary(X15FA)
X15FA1 <- dummy.data.frame(X15FA, names = NULL)
hist(X15FA$CompositeScore, main = "Composite Score Distribution", xlab = "Composite Score", ylab = "Number of Students")
hist(X15FA$DistanceFromCampus, main = "Distance From Campus Distribution", xlab = "Distance (miles)", ylab = "Number of Students")

#Clean the data, removing or replacing NA's
which((is.na(X15FA$HSGPA)))
X15FA$HSGPA[c(318,915,1967,2563,2669,3209)] <- median(X15FA$HSGPA, na.rm = TRUE)

which(is.na(X15FA$DistanceFromCampus))
X15FA$DistanceFromCampus[c(4,32,38,99,291,318,367,552,736,799,904,915,958,1063,1191,1333,1363,1419,1471,1602,1676,1799,1967,1989,2018,2241,2250,2254,2329,2335,2393,2453,2593,2669,2763,2964,2965,3027,3254,3256,3288,3457)] <- max(X15FA$DistanceFromCampus, na.rm = TRUE)

X15FA$`HSRank%` <- NULL
summary(X15FA)

#Principle Component Analysis
sub <- subset(X15FA,select = -c(Index, Enroll))
pca <- prcomp(sub, scale. = T)
pca$rotation
dim(pca$x)
biplot(pca, scale = 0)
std_dev <- pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)
plot(prop_varex, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b")

#split the data into training & testing
ind <- sample(2, nrow(X15FA), replace = TRUE, prob = c(0.8,0.2))
train15FA <- X15FA[ind == 1,]
test15FA <- X15FA[ind==2,]


#build a linear regression model
train15FA$Enroll <- as.factor(train15FA$Enroll)
reg15FA <- lm(formula= Enroll ~ ., data = train15FA)
predLR <- predict.lm(reg15FA, newdata = test15FA, interval = "predict")
confusionMatrix(predLR, reference = X15FA$Enroll, positive = "1")

#bagging & boosting.
#bagging method
train15FA$Enroll <- as.factor(train15FA$Enroll)
test15FA$Enroll <- as.factor(test15FA$Enroll)
X15FA.bagg <- bagging(Enroll ~ ., data = train15FA, mfinal = 10)
X15FA.bagg$importance

#test
X15FA.predbagging <- predict.bagging(X15FA.bagg, newdata = test15FA)
X15FA.predbagging$confusion
X15FA.predbagging$error

#10-fold cross validation
X15FA.baggingcv <- bagging.cv(Enroll ~., v=10, data = X15FA, mfinal = 100)
X15FA.baggingcv$confusion
X15FA.baggingcv$error

#boosting method
set.seed(4321)
X15FA.boost <- boosting(Enroll ~., data = train15FA, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
X15FA.predboost <- predict.boosting(X15FA.boost, newdata = test15FA)
X15FA.predboost$confusion
X15FA.predboost$error

#boosting with cross validation
X15FA.boostcv <- boosting.cv(Enroll~., v=10, data = X15FA, mfinal=100)
X15FA.boostcv$confusion
X15FA.boostcv$error


#build a random forest.
set.seed(1234)
rf <- randomForest(as.factor(Enroll) ~ ., data=train15FA, importance=TRUE, ntree=2000)
varImpPlot(rf)
PredRF <- predict(rf, test15FA, type = "class")
confusionMatrix(table(test15FA$Enroll, PredRF))
min(rf$err.rate)

#conditional inference tree
library(party)
set.seed(1234)
CItree <- ctree(Enroll ~., data = train15FA)
CItree
predCI <- predict(CItree, test15FA)
confusionMatrix(table(test15FA$Enroll, predCI))


#SVM
SVM15FA <- svm(Enroll ~ ., data = train15FA, kernel = "radial", cost = 1, gamma = 1/ncol(train15FA))
summary(SVM15FA)
SVMpred <- predict(SVM15FA, test15FA[, !names(test15FA) %in% c("Enroll")])
svm.table <- table(SVMpred, test15FA$Enroll)
svm.table
classAgreement(svm.table)
confusionMatrix(svm.table)

plot(SVM15FA, train15FA, Enroll ~ ., slice = list(Enroll = 2))


#deep learning?
train15FA$EnrollYes <- train15FA$Enroll == 1
train15FA$EnrollNo <- train15FA$Enroll == 2
network <- neuralnet(EnrollYes + EnrollNo ~ HSType + DistanceFromCampus + HSGPA + Visit + RegisPosition + FAFSASubmission, data = train15FA, hidden = 3)
network$result.matrix
plot(network)
net.predict <- compute(network, test15FA[-17])$net.result
net.prediction <- c("EnrollYes", "EnrollNo")[apply(net.predict, 1, which.max)]
predict.table <- table(test15FA$Enroll, net.prediction)
predict.table


#start h20 instance
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
pathtodata <- paste0(normalizePath("D:/Practicum/Project Data/15FA/"),"/15FA Final Data.xlsx", sheet = "Final 2")
write.table(x=X15FA, file = pathtodata, row.names = F, col.names = T)
dat_h2o <- h2o.importFile(path = pathtodata, destination_frame = "X15FA")
h2o.describe(dat_h2o)


#model1 with a deep learning network (three layers of 50 nodes)
model1 <- h2o.deeplearning(x=1:9,
                           y=10,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model1

#model2 with three layers of 100 nodes
model2 <- h2o.deeplearning(x=1:9,
                           y=10,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(100,100,100),
                           epochs = 50,
                           nfolds = 10)
model2

#model3 with three layers of 50 nodes and a Tanh activation
model3 <- h2o.deeplearning(x=1:9,
                           y=10,
                           training_frame = dat_h2o,
                           activation = "TanhWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model3

#model4 with three layers of 50 nodes and a Tanh activation, no dropout
model4 <- h2o.deeplearning(x=1:9,
                           y=10,
                           training_frame = dat_h2o,
                           activation = "Tanh",
                           input_dropout_ratio = 0.2,
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model4