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


#split the data into training & testing
ind4 <- sample(2, nrow(Admits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- Admits[ind4 == 1,]
testAdmits <- Admits[ind4 == 2,]


#build a linear regression model
trainAdmits$Enroll <- as.factor(trainAdmits$Enroll)
regAdmits <- glm(Enroll ~ ., family = "binomial", data = trainAdmits)
summary(regAdmits)
predLR <- predict(regAdmits, newdata = testAdmits, type = "response")
testAdmits$Probability <- predLR
class <- predLR >0.25
table(testAdmits$Enroll,class)
