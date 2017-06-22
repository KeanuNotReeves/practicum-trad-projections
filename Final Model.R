library(h2o)
library(readxl)
library(randomForest)
library(caret)
library(xlsx)
library(DMwR)
library(adabag)
library(e1071)
library(party)

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

#set variable types for PCA
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

#Principle Component Analysis
sub <-subset(numAdmits,select = -c(ID, Enroll))
pca <- prcomp(sub, scale. = T)
pca$rotation
dim(pca$x)
biplot(pca, scale = 0)
std_dev <- pca$sdev
pca_var <- std_dev^2
prop_varex <- pca_var/sum(pca_var)
plot(prop_varex, main = "Scree Plot", xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")
plot(cumsum(prop_varex), main = "Cumulative Sum of Variance", xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", type = "b")

PrimComps <- subset(Admits, select = c(Rating, FA_Intent, State, Visit, Time_between_App_and_Term, Regis_Position, Enroll))
PrimComps$Rating <- as.factor(PrimComps$Rating)
PrimComps$State <- as.factor(PrimComps$State)
PrimComps$Visit <- as.factor(PrimComps$Visit)
PrimComps$FA_Intent <- as.factor(PrimComps$FA_Intent)


#################################################################################################

Admits2 <- SMOTE(Enroll ~ ., Admits, perc.over = 500)
PrimComps2 <- SMOTE(Enroll ~ ., PrimComps, perc.over = 500)

#boosting with cross validation on all Admits
set.seed(2)
Admits.boostcv <- boosting.cv(Enroll~., v=10, data = Admits, mfinal=100)
Admits.boostcv$confusion
Admits.boostcv$error
Admits$Probability <- Admits.boostcv$class

#boosting with cross validation on Admits with balanced classes
set.seed(3)
Admits2.boostcv <- boosting.cv(Enroll~., v=10, data = Admits2, mfinal=100)
Admits2.boostcv$confusion
Admits2.boostcv$error
Admits2$Probability <- Admits2.boostcv$class
#write.xlsx(Admits2, "D:/Practicum/Project Data/Final.xlsx")

#boosting with cross validation on the PCA dataset
set.seed(4)
PCA.boostcv <- boosting.cv(Enroll~., v=10, data = PrimComps, mfinal=100)
PCA.boostcv$confusion
PCA.boostcv$error
PrimComps$Probability <- PCA.boostcv$class

#boosting with cross validation on the PCA dataset with balanced classes
set.seed(5)
PCA.SMOTE.boostcv <- boosting.cv(Enroll~., v=10, data = PrimComps2, mfinal=100)
PCA.SMOTE.boostcv$confusion
PCA.SMOTE.boostcv$error
PrimComps2$Probability <- PCA.SMOTE.boostcv$class

#################################################################################################
#break states into "in-state" and "out-of-state" to simplify the model
PrimComps2$State <- as.character(PrimComps2$State)
PrimComps2$State[PrimComps2$State == "CO"] <- "In"
PrimComps2$State[PrimComps2$State != "In"] <- "Out"
PrimComps2$State <- as.factor(PrimComps2$State)

#split the data into training & testing with Principle Components only
Prin <- sample(2, nrow(PrimComps2), replace = TRUE, prob = c(0.7,0.3))
trainPrin <- PrimComps2[Prin == 1,]
testPrin <- PrimComps2[Prin==2,]

#build a linear regression model
trainPrin$Enroll <- as.factor(trainPrin$Enroll)
set.seed(1)
regPrin <- glm(Enroll ~ ., family = "binomial", data = trainPrin)
summary(regPrin)
predLR <- predict(regPrin, newdata = testPrin, type = "response")
testPrin$Probability <- predLR
class <- predLR >0.35
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
                            hidden = c(200,200,200),
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
