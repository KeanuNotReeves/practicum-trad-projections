# **Traditional Student Enrollment Projections**    

  For this project I will use historical application information for Regis University to build a machine learning algorithm to assign a probability to each incoming student of future enrollment.The project was completed using Excel and R.

  In order to complete this I will pull all of the admitted students for the Fall term for the past two academic years (n = 7,435). I will pull in each of their academic credentials, as well as biographic and demographic to try and build the algorithm upon.

### **Data Collection**

  I acquired data from Regis University's data warehouse in accordance with the parameters of my job responsibilities and limitations. All of the personally identifying information of the student records was altered or deleted to protect their identity in accordance with FERPA guidelines.    


### **The Data**    

  The data has 7,435 samples and 17 features (12 categorical, 4 numeric, 1 identifier) including:

**ID**: A unique identifier of every student (not their actual student ID).

**Gender**: Male (M) or Female (F)

**Ethnic**: The student's self identified ethnicity (AN = American-Indian/Alaska Native, AS = Asian, BL = Black/African-American,                     HIS = Hispanic, HP = Hawaiian/Pacific Islander, Multiple = Multiple Ethnicities Reported, NR = Non-Resident Alien,                     Unknown = Unknown, WH = White)

**Religion**: The student's self identified religious belief. (BP = Baptist, BU = Buddhist, EP = Episcopalian, GO = Greek Orthodox,                     HU = Hindu, IS = Islam, JW = Jewish, LD = Latter Day Saints, LU = Lutheran, ME = Methodist, NA = Not Applicable,                           OP = Other Protestant, OT = Other, PB = Protestant, RC = Roman Catholic, UN = Unknown)    

** FA_Intent**: Yes, if the student indicated they wanted to submit financial aid paperwork to Regis University.

**First_Gen**: Yes, if the student is the first member of their family to attend college.

**HS_Type**: The type of high school the student attended.

**Distance**: The calculated distance from campus based on the zipcode of the student.

**State**: The residence state of the student.

**Composite Score**: A calculated score to translate ACT & SAT scores to a common scale.

**GPA**: The student's high school GPA.

**Rating**: The calculated rating of a student's desirability. (1 is lowest, 5 is highest).

**Visit**: Yes, if the student visited campus.

**Legacy**: Yes, if the student is a legacy at Regis University.

**Regis_Position**: The ranking the student gave to Regis University for colleges they would like to attend from their FAFSA file.

**Time_between_App_and_Term**: The calculated number of days between a student's application submissions and the beginning of the term.

**Enroll**: The target variable. Yes, if the student ended up enrolling at Regis University.    

### **EDA (Exploratory Data Analysis)**  

  EDA was performed by looking at the structure of the model and summary statistics. This showed a number of missing values that needed to be cleaned. 

```R
str(Admits)
summary(Admits)
hist(Admits$Composite_Score, main = "Composite Score Distribution", xlab = "Composite Score", 
     ylab = "Number of Students")
hist(Admits$Distance, main = "Distance From Campus Distribution", xlab = "Distance (miles)", 
     ylab = "Number of Students")
     
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
```

### **Analysis methods**

  In an effort to optimize the model as much as possible, I evaluated a number of models looking to optimize positive class recall. Positive class recall was selected as the criteria for verification because of the nature of the problem we are trying to solve. In assigning a probability to students of future enrollment, we are only hoping to help sort the list, not to definitively predict enrollment numbers. To that end, we want to maximize recall to make sure that the people who have a high probability, really are likely to enroll. Overall model accuracy does't matter due to the inherit imbalance of the classes. Class precision is also less important because if we are projecting more peole, this won't tell them we have a large class, rather that we have a lot of students who seem to want to come to Regis.    

I evaluated the following models:

1. Neural Network (h2o)
2. Conditional Inference Tree (randomForest)
3. Support Vector Machine (caret)
4. Bagging (adabag)
5. Boosting (adabag)
6. Logistic Regression (caret)

### **Results**

  To begin the project I used the raw data to train the models and output their class recall and overall accuracies. The table below shows how each model performed on the data.

![model](https://user-images.githubusercontent.com/17519823/27606569-12fcf7d0-5b3e-11e7-8d9b-6f0cb1c33e0f.png)

  You can see that while each model's accuracy is relatively high, the positive class recall is low. This is the baseline we are trying to improve upon. The first step in model improvement, was to dive in and see what may cause high accuracy and low recall. In my dataset this seems to be caused primarily by an imbalance in the target variable class (with a 6:1 ratio), and perhaps too many variables. 

![class imbalance](https://user-images.githubusercontent.com/17519823/27609674-d85fcd68-5b48-11e7-9b58-9cfe7c8e947b.png)


#### **SMOTE**    
  To try to remedy the class imbalance, I implemented a method called Synthetic Minority Oversampling Technique (SMOTE, from the DMwR package). SMOTE employs a k-Nearest Neighbor algorithm to build clusters of similar cases. Using these clusters, it creates synthetic cases in the minority class (in this case Enroll = "Yes") to balance the target variable. This then challenges the model to not just predict everyone as a "No", thus achieving an accuracy of 84%, but rather challenges it to actually look for patterns to predict a "Yes" correctly.    

```R
library(DMwR)
Admits2 <- SMOTE(Enroll ~ ., Admits, perc.over = 500)
```
![smote](https://user-images.githubusercontent.com/17519823/27609865-87292704-5b49-11e7-98ea-8695e1bf9ce4.png)

#### **PCA**    
  In addition, I also employed a principle component analysis (PCA) technique to try and identify variables that may be describing the same variance in the dataset. (The data had to be converted into all numeric values to complete the PCA). The biplot shows us that a lot of factors share vectors, meaning that they explain the same type of variance. The scree plot shows us that in order to successfully explain 90% of the variance in our data, we only need to use about 5 components. The PCA rotation shows us which 5 components to select, and from this we ended up with a dataset containing 7 features including : **Rating**, **FA Intent**, **State**, **Visit**, **Time between App and Term**, **Regis Position** and **Enroll**.

```R
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
```

![biplot](https://user-images.githubusercontent.com/17519823/27610211-a7392ff2-5b4a-11e7-9501-f6763ccca56a.png)

![scree](https://user-images.githubusercontent.com/17519823/27610309-0dcdf360-5b4b-11e7-89f9-53e24266e23d.png)

#### **Improved Models**
  Using the methods outlined above, I retrained some models to see how using SMOTE and PCA would improve class recall. The table below illustrates the model improvements before and after each method was applied. The table clearly shows that using the SMOTE method dramatically increased the class recall of every model, more so that the PCA did alone.

![more models](https://user-images.githubusercontent.com/17519823/27708286-a688b848-5cd5-11e7-949f-68264eab99bc.png)

  In the end, using Principle Component Analysis did not help with modeling as much as the SMOTE technique did. Using SMOTE to balance the classes, we can use either the boosting, or regression algorithm we can build a model off of previous years data to assign a probability of enrollment to new, unscored, data. THe final code to compete this task is below. 
  
```R
library(readxl)
library(DMwR)
library(adabag)
library(e1071)
library(xlsx)

######################## bring in the 15FA & 16FA data ###########################
X15FA <- as.data.frame(read_excel("D:/Practicum/Project Data/15FA/15FA Final.xlsx",sheet = "15FA"))
X16FA <- as.data.frame(read_excel("D:/Practicum/Project Data/16FA/16FA Final.xlsx",sheet = "Final"))
X17FA <- as.data.frame(read_excel("D:/Practicum/Project Data/17FA/17FA Final.xlsx",sheet = "Final"))

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
X17FA$Enroll <- as.factor(X17FA$Enroll)

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

################# use SMOTE to balance the classes ########################

SMOTEAdmits <- SMOTE(Enroll ~ ., Admits, perc.over = 500)

############ split the Admits SMOTE data into training & testing ################
ind <- sample(2, nrow(SMOTEAdmits), replace = TRUE, prob = c(0.8,0.2))
trainAdmits <- SMOTEAdmits[ind == 1,]
testAdmits <- SMOTEAdmits[ind == 2,]

############# Boosting with the SMOTE method applied on the dataset ###################
set.seed(1234)
Admits.boost <- boosting(Enroll ~., data = SMOTEAdmits, mfinal=10, coeflearn = "Breiman", control = rpart.control(maxdepth = 3))
Admits.predboost <- predict.boosting(Admits.boost, newdata = X17FA)
X17FA$Probabiliy <- Admits.predboost$prob[,2]

write.xlsx(X17FA, "D:/Practicum/Project Data/17FA/17FA_Output.xlsx", sheet = "Output")
```

  The final command outputes the newly scored data back into an Excel spreadsheet so it can be imported into the admissions CRM system for Admission Counselor use.

### **Summary**

  The working model created by the analysis is a great first step, and in many ways is Regis University's first foray into the realm of Data Science. The model should be implemented in January 2018, and with that we are hoping to use it to highlight certain ways that we can expand upon our data collection practices to build a better model in the future.     

  What this initial model really highlighted was that if we want to predict who is coming to Regis, we need to look at any data points that reflect how a student feels about Regis University. Test scores, ethnicity, state; none of these things really mattered in the model because they really tell us nothing about the student. What told us everything was: visit, FA_intent, Regis Position and the time between the term and their app being submitted. These data points reflect a student's feelings towards us, and therefore better predict if a student will come here. 

  To continue to build upon the model, we hope that showing this to various groups at Regis Universty will propel them to try and collect more data that can provide insight into a student's feelings towards the university. Things like how often the student contacts the their Admissions Counselor, or visits our landing page are items at the top of the list for exploration.

### **References**

SMOTE - https://www.jair.org/media/953/live-953-2037-jair.pdf    
PCA - https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/    
Random Forest - http://trevorstephens.com/kaggle-titanic-tutorial/r-part-5-random-forests/    
H2o package - https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning    
Adabag package - https://cran.r-project.org/web/packages/adabag/adabag.pdf    


