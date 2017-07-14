library(readxl)
library(DMwR)
library(adabag)
library(e1071)
library(xlsx)

######################## bring in the 15FA & 16FA data ###########################
X15FA <- as.data.frame(read_excel("W:/I-J/IAR_Institutional_Research/Predictive Analytics/Freshman Enrollment Probability/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))
X16FA <- as.data.frame(read_excel("W:/I-J/IAR_Institutional_Research/Predictive Analytics/Freshman Enrollment Probability/Practicum/Project Data/16FA/16FA Final.xlsx",sheet = "Final"))
X17FA <- as.data.frame(read_excel("W:/I-J/IAR_Institutional_Research/Predictive Analytics/Freshman Enrollment Probability/Practicum/Project Data/17FA/17FA Final.xlsx",sheet = "Final"))

Admits <- rbind(X15FA, X16FA)

################### set variable types ##########################
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

############## Clean the data, removing or replacing NA's #######################
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

Admits$State <- as.character(Admits$State)
Admits$State[Admits$State == "CO"] <- "In"
Admits$State[Admits$State != "In"] <- "Out"
Admits$State <- as.factor(Admits$State)

X17FA$State <- as.character(X17FA$State)
X17FA$State[X17FA$State == "CO"] <- "In"
X17FA$State[X17FA$State != "In"] <- "Out"
X17FA$State <- as.factor(X17FA$State)

which(X17FA$Rating==0)
X17FA$Rating[c(4333,4342,4359,4360,4387)] <- 1

################# use SMOTE to balance the classes ########################
SMOTEAdmits <- SMOTE(Enroll ~ ., Admits, perc.over = 500)

################# use PCA to simplify data########################
PrimComps <- subset(SMOTEAdmits, select = c(Rating, FA_Intent, State, Visit, Time_between_App_and_Term, Regis_Position, Enroll))
PrimComps$Rating <- as.factor(PrimComps$Rating)
PrimComps$State <- as.factor(PrimComps$State)
PrimComps$Visit <- as.factor(PrimComps$Visit)
PrimComps$FA_Intent <- as.factor(PrimComps$FA_Intent)

PCA17FA <- subset(X17FA, select = c(ID, Rating, FA_Intent, State, Visit, Time_between_App_and_Term, Regis_Position))

############ split the Admits SMOTE data into training & testing ################
ind <- sample(2, nrow(PrimComps), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- PrimComps[ind == 1,]
testAdmits <- PrimComps[ind == 2,]

############# Regression with the SMOTE and PCA method applied on the dataset ###################
#set.seed(5678)
#regAdmits <- glm(Enroll ~ ., family = "binomial", data = trainAdmits)
#summary(regAdmits)
#predAdmits <- predict(regAdmits, newdata = testAdmits, type = "response")
#class <- predAdmits > 0.30
#table(testAdmits$Enroll,class)
#testAdmits$Probability <- predAdmits

set.seed(4562)
reg17FA <- glm(Enroll ~ ., family = "binomial", data = PrimComps)
summary(reg17FA)
pred17FA <- predict(reg17FA, newdata = PCA17FA, type = "response")
PCA17FA$Probability <- pred17FA
summary(PCA17FA$Probability)
write.xlsx(PCA17FA, "W:/I-J/IAR_Institutional_Research/Predictive Analytics/Freshman Enrollment Probability/Practicum/Project Data/17FA/17FA_Output.xlsx", sheet = "Output")
