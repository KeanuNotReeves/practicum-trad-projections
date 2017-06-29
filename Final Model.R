library(readxl)
library(DMwR)
library(adabag)
library(e1071)
library(xlsx)

#bring in the 15FA & 16FA data
X15FA <- as.data.frame(read_excel("D:/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))
X16FA <- as.data.frame(read_excel("D:/Practicum/Project Data/16FA/16FA Final.xlsx",sheet = "Final"))
X17FA <- as.data.frame(read_excel("D:/Practicum/Project Data/17FA/17FA Final.xlsx",sheet = "Final"))

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

X17FA$Gender <- as.factor(X17FA$Gender)
X17FA$Ethnic <- as.factor(X17FA$Ethnic)
X17FA$Religion <- as.factor(X17FA$Religion)
X17FA$FA_Intent <- as.factor(X17FA$FA_Intent)
X17FA$First_Gen <- as.factor(X17FA$First_Gen)
X17FA$HS_Type <- as.factor(X17FA$HS_Type)
X17FA$State <- as.factor(X17FA$State)
X17FA$Rating <- as.factor(X17FA$Rating)
X17FA$Visit <- as.factor(X17FA$Visit)
X17FA$Legacy <- as.factor(X17FA$Legacy)
X17FA$Regis_Position <- as.factor(X17FA$Regis_Position)
X17FA$Enroll <- as.factor(X17FA$Enroll)

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

X17FA$GPA[is.na(X17FA$GPA)] <- with(X17FA, median(X17FA$GPA, na.rm = TRUE))
X17FA$Distance[is.na(X17FA$Distance)] <- with(X17FA, max(X17FA$Distance, na.rm = TRUE))
X17FA$HS_Type[is.na(X17FA$HS_Type)] <- "HO"

X17FA$State <- as.character(X17FA$State)
X17FA$State[is.na(X17FA$State)] <- "International"
X17FA$State <- as.factor(X17FA$State)

#break states into "in-state" and "out-of-state" to simplify the model
#Admits$State <- as.character(Admits$State)
#Admits$State[Admits$State == "CO"] <- "In"
#Admits$State[Admits$State != "In"] <- "Out"
#Admits$State <- as.factor(Admits$State)

SMOTEAdmits <- SMOTE(Enroll ~ ., Admits, perc.over = 500)

#################################################################################################
################################################################################################
############# Boosting with the SMOTE method applied on the dataset #############

#split the Admits SMOTE data into training & testing
ind <- sample(2, nrow(SMOTEAdmits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- SMOTEAdmits[ind == 1,]
testAdmits <- SMOTEAdmits[ind == 2,]

#boosting method with balanced classes
set.seed(1234)
Admits.boost <- boosting(Enroll ~., data = SMOTEAdmits, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Admits.predboost <- predict.boosting(Admits.boost, newdata = X17FA)
X17FA$Probabiliy <- Admits.predboost$prob[,2]

write.xlsx(X17FA, "D:/Practicum/Project Data/17FA/17FA_Output.xlsx", sheet = "Output")
