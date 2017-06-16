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

#split the data into training & testing based on numerical data frame
ind2 <- sample(2, nrow(numAdmits), replace = TRUE, prob = c(0.8,0.2))
trainNum <- numAdmits[ind2 == 1,]
testNum <- numAdmits[ind2 == 2,]

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
