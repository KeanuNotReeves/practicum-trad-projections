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
