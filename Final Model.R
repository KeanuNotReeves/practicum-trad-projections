library(h2o)
library(readxl)
library(randomForest)
library(caret)
library(xlsx)
library(DMwR)
library(adabag)
library(e1071)
library(party)
library(corrplot)

#bring in the 15FA & 16FA data
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

#create a numeric dataset for correlations and PCA
numAdmits <- Admits
numAdmits$Gender <- as.numeric(numAdmits$Gender)
numAdmits$Ethnic <- as.numeric(numAdmits$Ethnic)
numAdmits$Religion <- as.numeric(numAdmits$Religion)
numAdmits$FA_Intent <- as.numeric(numAdmits$FA_Intent)
numAdmits$First_Gen <- as.numeric(numAdmits$First_Gen)
numAdmits$HS_Type <- as.numeric(numAdmits$HS_Type)
numAdmits$Visit <- as.numeric(numAdmits$Visit)
numAdmits$Rating <- as.numeric(numAdmits$Rating)
numAdmits$Legacy <- as.numeric(numAdmits$Legacy)
numAdmits$Enroll <- as.numeric(numAdmits$Enroll)
numAdmits$Regis_Position <- as.numeric(numAdmits$Regis_Position)
numAdmits$State <- as.numeric(numAdmits$State)

#perform EDA on the dataset
str(Admits)
summary(Admits)
hist(Admits$Composite_Score, main = "Composite Score Distribution", xlab = "Composite Score", 
     ylab = "Number of Students")
hist(Admits$Distance, main = "Distance From Campus Distribution", xlab = "Distance (miles)", 
     ylab = "Number of Students")
corr <- cor(numAdmits, method = "pearson", use = "complete.obs")
corrplot(corr, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

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


#Principle Component Analysis
sub <-subset(numAdmits,select = -c(ID, Enroll))
pca <- prcomp(sub, scale. = T)
pca$rotation
dim(pca$x)
biplot(pca, scale = 0)
std_dev <- pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)
plot(prop_varex, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", type = "b")
plot(cumsum(prop_varex), main = "Cumulative Sum of Variance", 
     xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b")

PrimComps <- subset(Admits, select = c(Rating, FA_Intent, State, Visit, Time_between_App_and_Term, Regis_Position, Enroll))
PrimComps$Rating <- as.factor(PrimComps$Rating)
PrimComps$State <- as.factor(PrimComps$State)
PrimComps$Visit <- as.factor(PrimComps$Visit)
PrimComps$FA_Intent <- as.factor(PrimComps$FA_Intent)

#Remedy imbalanced classes
fold <- table(Admits$Enroll)
colors <- c("#F1C400","#002B49")
pie(fold, main = "Enrolled Status for All Admits", col = colors)
Admits2 <- SMOTE(Enroll ~ ., Admits, perc.over = 500)
PrimComps2 <- SMOTE(Enroll ~ ., PrimComps, perc.over = 500)

#################################################################################################
#################################################################################################

#split the Admits data into training & testing
ind <- sample(2, nrow(Admits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- Admits[ind == 1,]
testAdmits <- Admits[ind == 2,]

#boosting method on all Admits
set.seed(1)
Admits.boost <- boosting(Enroll ~., data = trainAdmits, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Admits.predboost <- predict.boosting(Admits.boost, newdata = testAdmits)
Admits.predboost$confusion
Admits.predboost$error

#boosting with cross validation on all Admits
set.seed(2)
Admits.boostcv <- boosting.cv(Enroll~., v=10, data = Admits, mfinal=100)
Admits.boostcv$confusion
Admits.boostcv$error
Admits$Probability <- Admits.boostcv$class

#split the data into training & testing with Principle Components only
Prin <- sample(2, nrow(PrimComps), replace = TRUE, prob = c(0.7,0.3))
trainPrin <- PrimComps[Prin == 1,]
testPrin <- PrimComps[Prin==2,]

#boosting method on Principle Components
set.seed(3)
Prim.boost <- boosting(Enroll ~., data = trainPrin, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Prim.predboost <- predict.boosting(Prim.boost, newdata = testPrin)
Prim.predboost$confusion
Prim.predboost$error

#boosting with cross validation on the PCA dataset
set.seed(4)
PCA.boostcv <- boosting.cv(Enroll~., v=10, data = PrimComps, mfinal=100)
PCA.boostcv$confusion
PCA.boostcv$error
PrimComps$Probability <- PCA.boostcv$class


############# Boosting with the SMOTE method applied on the dataset #############

#split the Admits SMOTE data into training & testing
ind2 <- sample(2, nrow(Admits2), replace = TRUE, prob = c(0.8,0.2))
trainAdmits2 <- Admits2[ind2 == 1,]
testAdmits2 <- Admits2[ind2 == 2,]

#boosting method with balanced classes
set.seed(5)
Admits2.boost <- boosting(Enroll ~., data = trainAdmits2, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Admits2.predboost <- predict.boosting(Admits2.boost, newdata = testAdmits2)
Admits2.predboost$confusion
Admits2.predboost$error

#boosting with cross validation on Admits with balanced classes
set.seed(6)
Admits2.boostcv <- boosting.cv(Enroll~., v=10, data = Admits2, mfinal=100)
Admits2.boostcv$confusion
Admits2.boostcv$error
Admits2$Probability <- Admits2.boostcv$class
#write.xlsx(Admits2, "D:/Practicum/Project Data/Final.xlsx")

#split the PCA SMOTE data into training & testing
ind4 <- sample(2, nrow(PrimComps2), replace = TRUE, prob = c(0.8,0.2))
trainPrim2 <- PrimComps2[ind4 == 1,]
testPrim2 <- PrimComps2[ind4 == 2,]

#boosting method on Principle Components
set.seed(7)
Prim2.boost <- boosting(Enroll ~., data = trainPrim2, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Prim2.predboost <- predict.boosting(Prim2.boost, newdata = testPrim2)
Prim2.predboost$confusion
Prim2.predboost$error

#boosting with cross validation on the PCA dataset with balanced classes
set.seed(6)
PCA.SMOTE.boostcv <- boosting.cv(Enroll~., v=10, data = PrimComps2, mfinal=100)
PCA.SMOTE.boostcv$confusion
PCA.SMOTE.boostcv$error
PrimComps2$Probability <- PCA.SMOTE.boostcv$class

#################################################################################################
#################################################################################################

#break states into "in-state" and "out-of-state" to simplify the model
Admits$State <- as.character(Admits$State)
Admits$State[Admits$State == "CO"] <- "In"
Admits$State[Admits$State != "In"] <- "Out"
Admits$State <- as.factor(Admits$State)

ind5 <- sample(2, nrow(Admits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- Admits[ind5 == 1,]
testAdmits <- Admits[ind5 == 2,]

#build a linear regression model on all Admits
trainAdmits$Enroll <- as.factor(trainAdmits$Enroll)
set.seed(7)
regAdmits <- glm(Enroll ~ ., family = "binomial", data = trainAdmits)
summary(regAdmits)
predAdmits <- predict(regAdmits, newdata = testAdmits, type = "response")
#testAdmits$Probability <- predAdmits
class <- predAdmits >0.35
table(testAdmits$Enroll,class)

#break states into "in-state" and "out-of-state" to simplify the model
PrimComps$State <- as.character(PrimComps$State)
PrimComps$State[PrimComps$State == "CO"] <- "In"
PrimComps$State[PrimComps$State != "In"] <- "Out"
PrimComps$State <- as.factor(PrimComps$State)


#build a linear regression model on PCA
trainPrin$Enroll <- as.factor(trainPrin$Enroll)
set.seed(7)
regPrin <- glm(Enroll ~ ., family = "binomial", data = trainPrin)
summary(regPrin)
predLR <- predict(regPrin, newdata = testPrin, type = "response")
#testPrin$Probability <- predLR
class <- predLR >0.35
table(testPrin$Enroll,class)


#break states into "in-state" and "out-of-state" to simplify the model
Admits$State <- as.character(Admits$State)
Admits$State[Admits$State == "CO"] <- "In"
Admits$State[Admits$State != "In"] <- "Out"
Admits$State <- as.factor(Admits$State)

############################## SMOTE ################################
#build a linear regression model on all Admits with balanced classes

trainAdmits2$Enroll <- as.factor(trainAdmits2$Enroll)
set.seed(7)
regAdmits2 <- glm(Enroll ~ ., family = "binomial", data = trainAdmits2)
summary(regAdmits2)
predAdmits2 <- predict(regAdmits2, newdata = testAdmits2, type = "response")
#testAdmits$Probability <- predAdmits
class <- predAdmits2 >0.35
table(testAdmits2$Enroll,class)

#break states into "in-state" and "out-of-state" to simplify the model for balanced classes
PrimComps2$State <- as.character(PrimComps2$State)
PrimComps2$State[PrimComps2$State == "CO"] <- "In"
PrimComps2$State[PrimComps2$State != "In"] <- "Out"
PrimComps2$State <- as.factor(PrimComps2$State)

#split the data into training & testing with Principle Components only and balanced classes
Prin2 <- sample(2, nrow(PrimComps2), replace = TRUE, prob = c(0.7,0.3))
trainPrin2 <- PrimComps2[Prin == 1,]
testPrin2 <- PrimComps2[Prin==2,]

#build a linear regression model with principle components & balanced classes
trainPrin2$Enroll <- as.factor(trainPrin2$Enroll)
set.seed(8)
regPrin2 <- glm(Enroll ~ ., family = "binomial", data = trainPrin2)
summary(regPrin2)
predLR2 <- predict(regPrin2, newdata = testPrin, type = "response")
#testPrin2$Probability <- predLR2
class <- predLR2 >0.35
table(testPrin$Enroll,class)

###################################################################################################

#Neural network with H2O package
#start h20 instance

h2oPCA <- as.h2o(PrimComps2, destination_frame = "PrimComps2")
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
dim(h2oPCA)
h2oPCA
splits <- h2o.splitFrame(h2oPCA, c(0.6,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%


#model6 with a deep learning network (three layers of 50 nodes)
dlmodel <- h2o.deeplearning(x=1:6,
                            y=7,
                            set.seed(789),
                            training_frame = train,
                            validation_frame = test,
                            activation = "RectifierWithDropout",
                            input_dropout_ratio = 0.1,
                            hidden_dropout_ratios = c(0.5,0.5,0.5),
                            hidden = c(50,50,50),
                            epochs = 100,
                            nfolds = 10,
                            variable_importances = TRUE,
                            overwrite_with_best_model = TRUE,
                            balance_classes = TRUE)
dlmodel
head(as.data.frame(h2o.varimp(dlmodel)))

#h2o.performance(dlmodel, train = T)
h2o.performance(dlmodel, valid = T)
#h2o.performance(dlmodel, newdata = train)
h2o.performance(dlmodel, newdata = valid)
h2o.performance(dlmodel, newdata = test)

DLpred <- h2o.predict(dlmodel, test)
DLpred
test$Accuracy <- DLpred$predict == test$Enroll
1-mean(test$Accuracy)
plot(dlmodel)
