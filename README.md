# **Traditional Student Enrollment Projections**    

For this project I will use historical application information for Regis University to build a machine learning algorithm to assign a probability to each incoming student of future enrollment.

In order to complete this I will pull all of the admitted students for the Fall term for the past two academic years (approximately 7,000 stdents). I will pull in each of their academic credentials, as well as biographic and demographic to try and build the algorithm upon.

### **Data Collection**

I acquired data from Regis University's data warehouse in accordance with the parameters of my job responsibilities and limitations. All of the personally identifying information of the student records was altered or deleted to protect their identity in accordance with FERPA guidelines.    


#### **The Data**    

The data has 7,000 samples and 16 features including ID, Gender, Ethnic, Religion, First Generation, High School Type, Distance, State, Composite Score, High School GPA, Rating, Visit, Legacy, Regis Position, Time between Application Submitted and Term Start, and Enroll. The data has 11 categorical variables and 5 numeric. The target variable is Enroll, which is a factor with two levels ("Yes" and "No").    

### **EDA (Exploratory Data Analysis)**    

### **Analysis methods**

Describe your analysis methods and brief explanations on your assumption. Include figures or tables as needed.
When writing analysis method, also explicitly mention which tools and libraries you used (imagine this is your portfolio to your future employer- you want to be as impressive as possible) Results from your analysis.

### **Results**

You can make model(s) comparison if you have multiple approaches. You may include any tables or graphs that summarizes your result(s). Your conclusion, any discussions on difficulties (why some approach didn't work), and any ideas how you may overcome those. If you have external demo website, you can include a link your webapp.

### **Summary**

Summarize your work and highlight your takeaways. You may include ideas on how you may improve the analysis, or future steps.

### **References**


#### **Variables Explained**

ID: A unique identifier of every student (not their actual student ID).

Gender: Male (M) or Female (F)

Ethnic: The student's self identified ethnicity (AN = American-Indian/Alaska Native, AS = Asian, BL = Black/African-American,                     HIS =     Hispanic, HP = Hawaiian/Pacific Islander, Multiple = Multiple Ethnicities Reported, NR = Non-Resident Alien,                     Unknown = Unknown, WH = White)

Religion: The student's self identified religious belief. (BP = Baptist, BU = Buddhist, EP = Episcopalian, GO = Greek Orthodox,                     HU = Hindu, IS = Islam, JW = Jewish, LD = Latter Day Saints, LU = Lutheran, ME = Methodist, NA = Not Applicable,                           OP = Other Protestant, OT = Other, PB = Protestant, RC = Roman Catholic, UN = Unknown)

First_Gen: Yes, if the student is the first member of their family to attend college.

HS_Type: The type of high school the student attended.

Distance: The calculated distance from campus based on the zipcode of the student.

State: The residence state of the student.

Composite Score: A calculated score to translate ACT & SAT scores to a common scale.

GPA: The student's high school GPA.

Rating: The calculated rating of a student's desirability. (1 is lowest, 5 is highest).

Visit: Yes, if the student visited campus.

Legacy: Yes, if the student is a legacy at Regis University.

Regis_Position: The ranking the student gave to Regis University for colleges they would like to attend from their FAFSA file.

Time_between_App_and_Term: The calculated number of days between a student's application submissions and the beginning of the term.

Enroll: The target variable. Yes, if the student ended up enrolling at Regis University.
