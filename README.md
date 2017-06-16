# practicum-trad-projections
For this project I will use historical application information for Regis University to build a machine learning algorithm to assign a probability to each incoming student of future enrollment.

In order to complete this I will pull all of the admitted for the Fall term for the past two academic years (approximately 7,000 stdents). I will pull in each of their academic credentials, as well as biographic and demographic to try and build the algorithm upon. The variables are as follows:

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
