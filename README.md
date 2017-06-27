# **Traditional Student Enrollment Projections**    

For this project I will use historical application information for Regis University to build a machine learning algorithm to assign a probability to each incoming student of future enrollment.The project was completed using Excel and R.

In order to complete this I will pull all of the admitted students for the Fall term for the past two academic years (approximately 7,000 stdents). I will pull in each of their academic credentials, as well as biographic and demographic to try and build the algorithm upon.

### **Data Collection**

I acquired data from Regis University's data warehouse in accordance with the parameters of my job responsibilities and limitations. All of the personally identifying information of the student records was altered or deleted to protect their identity in accordance with FERPA guidelines.    


### **The Data**    

The data has 7,000 samples and 16 features including:

**ID**: A unique identifier of every student (not their actual student ID).

**Gender**: Male (M) or Female (F)

**Ethnic**: The student's self identified ethnicity (AN = American-Indian/Alaska Native, AS = Asian, BL = Black/African-American,                     HIS =     Hispanic, HP = Hawaiian/Pacific Islander, Multiple = Multiple Ethnicities Reported, NR = Non-Resident Alien,                     Unknown = Unknown, WH = White)

**Religion**: The student's self identified religious belief. (BP = Baptist, BU = Buddhist, EP = Episcopalian, GO = Greek Orthodox,                     HU = Hindu, IS = Islam, JW = Jewish, LD = Latter Day Saints, LU = Lutheran, ME = Methodist, NA = Not Applicable,                           OP = Other Protestant, OT = Other, PB = Protestant, RC = Roman Catholic, UN = Unknown)

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
```

### **Analysis methods**

In an effort to optimize the model as much as possible, I evaluated a number of models looking to optimize positive class recall. Positive class recall was selected as the criteria for verification because of the nature of the problem we are trying to solve. In assigning a probability to students of future enrollment, we are only hoping to help sort the list, not to definitively predict enrollment numbers. To that end, we want to maximize recall to make sure that the people who have a high probability, really are likely to enroll. Overall model accuracy does't matter due to the inherit imbalance of the classes. Class precision is also less important because if we are projecting more peole, this won't tell them we have a large class, rather that we have a lot of students who seem to want to come to Regis.    

I evaluated the following models:

Neural Network (h2o)
Conditional Inference Tree (randomForest)
Support Vector Machine (caret)
Bagging (adabag)
Boosting (adabag)
Logistic Regression (caret)

### **Results**

To begin the project I used the raw data to train the models and output their class recall and overall accuracies. The table below shows how each model performed on the data.

![model](https://user-images.githubusercontent.com/17519823/27606569-12fcf7d0-5b3e-11e7-8d9b-6f0cb1c33e0f.png)

You can see that while each model's accuracy is relatively high, the positive class recall is low. This is the baseline we are trying to improve upon. The first step in model improvement, was to dive in and see what may cause high accuracy and low recall. In my dataset this seems to be caused primarily by an imbalance in the target variable class (with a 6:1 ratio), and perhaps too many variables. 

![class imbalance](https://user-images.githubusercontent.com/17519823/27609674-d85fcd68-5b48-11e7-9b58-9cfe7c8e947b.png)

To try to remedy the class imbalance, I implemented a method called Synthetic Minority Oversampling Technique (SMOTE, from the DMwR package). SMOTE employs a k-Nearest Neighbor algorithm to build clusters of similar cases. Using these clusters, it creates synthetic cases in the minority class (in this case Enroll = "Yes") to balance the target variable. This then challenges the model to not just predict everyone as a "No", thus achieving an accuracy of 84%, but rather challenges it to actually look for patterns to predict a "Yes" correctly.    

```R
library(DMwR)
Admits2 <- SMOTE(Enroll ~ ., Admits, perc.over = 500)
```
![smote](https://user-images.githubusercontent.com/17519823/27609865-87292704-5b49-11e7-98ea-8695e1bf9ce4.png)

In addition, I also employed a principle component analysis (PCA) technique to try and identify variables that may be describing the same variance in the dataset. 

![biplot](https://user-images.githubusercontent.com/17519823/27610211-a7392ff2-5b4a-11e7-9501-f6763ccca56a.png)

### **Summary**

The working model created by the analysis is a great first step, and in many ways is Regis University's first foray into the realm of Data Science. The model should be implemented in January 2018, and with that we are hoping to use it to highlight certain ways that we can expand upon our data collection practices to build a better model in the future.     

What this initial model really highlighted was that if we want to predict who is coming to Regis, we need to look at any data points that reflect how a student feel about Regis University. Test scores, ethnicity, state; none of these things really mattered in the model because they really tell us nothing about the student. What told us everything was: visit, FA_intent, Regis Position and the time between the term and their app being submitted. THese data points reflect a student's feelings towards us, and therefore better predict if a student will come here. 

### **References**

SMOTE - https://www.jair.org/media/953/live-953-2037-jair.pdf
