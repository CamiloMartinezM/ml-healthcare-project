string = """
2 Problem Description
In healthcare, effective allocation of limited resources is crucial to adequately address the needs of all patients
optimally. To better inform healthcare allocation, predictive analytics have become crucial in recent times to
predict the healthcare usage of individuals.
Your objective for this project is to develop machine learning models capable of forecasting patients’ health-
care utilization and the total cost incurred by each individual. To enable learning such models, you are provided
with a dataset compiled from a survey of patients in the United States. This dataset encompasses various fea-
tures related to health data, demographics, and specific services utilized by individuals.
The specific target features in this dataset and the respective tasks pertaining to these target features are
as follows.
1. Total Medical Expenditure: This is a continuous feature measuring the total healthcare expenditure
in US dollars. Your goal is to build a regression model that, when deployed, can take the features of
previously unseen patients as input and predict the total medical expenditure.
2. Healthcare Utilization: This is a binary categorical feature that can either indicate LOW usage or
HIGH usage. Your goal is to build a binary classification model that, when deployed, can take the features
of previously unseen patients as input and predict the utilization category of the healthcare system.
2.1 Datasets
We share three datasets with you, one training set and two test sets. For each set, the data will contain 108
features and, in the case of the training set, two target features. The targets are TOT MED EXP indicating the
continuous healthcare expenditure in US dollars and UTILIZATION indicating the binary utilization class (LOW
or HIGH). The data features include:
• Demographics: RACE, SEX, AGE, PANEL (survey panel number), REGION (Census region), MARITAL STAT
(marital status), POVRTY CAT (poverty category), POVRTY LEV (poverty level), EDU YRS (education years),
EDU DEG (highest education degree), SPOUSE PRSNT (marital status with spouse present), STUDENT STAT
(student status), UNION STAT (union status), NUM DEP OUT REP UNT (number of dependents outside survey
1
reporting unit), EMPLYMT (employment status), OCCUP (occupation), NON ENG LANG (non-English language
spoken).
• Personal: PUB ASST (public assistance money), TAX FORM TYP (tax form type submitted), FOOD STMP MNTHS
(number of months food stamps purchased), FOOD STMP VAL (value of food stamps), MIL ACTIV DUTY
(military active duty), HON DISCHARGE (honorary discharge from army), INSUR COV (insurance cover-
age), TOT INCOME (total income), EMPLYR INS (employer offers insurance), CHILD SUPP (child support),
PROB WKIDS (problem with kids), FAM INCOME (family income), PROB BILL PAY (problem with bill pay-
ments), DELAY PRESCT MED (delay getting prescription medication), DAYS CAREOTHR NOWORK (days not
working due to care for others), PENSN PLAN (pension plan), NO WORK WHY (reason for not working).
• Health-related: WEIGHT, HEALTH STAT (perceived health status), MENTAL HLTH (perceived mental health),
CHRON BRONCH (chronic bronchitis), JNT PAIN (joint pain), PREGNT (pregnant), WALK LIM (walking lim-
itation), ACTIV LIM (activity limitation), SOCIAL LIM (social limitation), COGNTV LIM (cognitive limi-
tation), BM IDX (BMI), MULT HIGHBP (multiple high blood pressure readings), HOUSEWRK LIM (house-
work limitation), SCHOOL LIM (school limitation), ADV NO FAT FOOD (advised to restrict high-fat food),
ADV EXERCISE MORE (advised to exercise), ADV DNTL CKP (advised dental checkup), FREQ DNTL CKP (fre-
quency of dental checkup), RSN NO DNTL CKP (reason for no dental checkup), RSN NO MED CKP (reason
for no medical checkup), DOC CK BP (doctor checked blood pressure), TAKE RISK (prone to taking risks),
ADV BOOST SEAT (advised booster seat), WHEN ADV BOOST SEAT (when advised booster seat), FEEL DEPRS
(feels depressed), ADV NO SMKG (advised no smoking), AGE DIAG ADHD (age when diagnosed ADHD), PROB WBHV
(problem with home behavior), WEAR SEATBLT (wear seat belt), WHEN ADV LAP BLT (when advised lap
belt), WHEN LST ASTHMA (when last asthma episode), ADV LAP BLT (advised lap belt), ADV EAT HLTHY
(advised to eat healthy), DOC TIM ALN (doctor spent time alone), APPT REG MEDCARE (made routine ap-
pointment medical care), LOST ALL TEETH, ASPRN REG (regular aspirin usage), DIFF ERRND ALN (difficulty
doing errands alone), DIAB KIDNY (diabetes-related kidney issue), DIAB INSLN (diabetes insulin use),
DIAB MED (diabetes medicine use), DISPSN STAT (patient disposition status), TIME LAST PSA (last PSA
test), WHEN ADV EXERCISE (when advised to exercise more), DEAF, BLIND, LAST FLU VAC (last flu vaccine),
UNABL PRES MED (unable to get proper medicine), HEAR AID (need hearing aid), LAST REG CKP (last regular
checkup), DAYS ILL NOWORK (days miss work for illness), DAYS ILL NOSCHL (days miss school for illness),
HIGH BP DIAG, COR HRT DIAG (coronary heart disease diagnosis), ANGINA DIAG, HRT ATT DIAG (heart attack
diagnosis), OTH HRT DIAG (other heart-related diagnosis), STROKE DIAG (stroke diagnosis), EMPHYM DIAG
(emphysema diagnosis), HIGHCHOL DIAG (high cholesterol diagnosis), CANCER DIAG, DIAB DIAG (diabetes),
ARTHR DIAG (arthritis), ARTHR TYPE (arthritis type), ASTHM DIAG (asthma), ADHD DIAG, NUM PRESCR MEDS
(number of prescription medicines), DIFFIC HEAR (difficulty hearing), DIFFIC SEE (difficulty seeing),
SMOK (smoking), OVR FEEL 14 (overall feeling rating 14 days), MENTAL HLTH SCR (mental health score),
PHY HLTH SCR (physical health score), OVR FEEL 30 (overall feeling rating 30 days).
The training data has information and labels for 15000 patients. The first test dataset, named test public.csv,
has features of 4791 patients without labels. You can access it right now on CMS. The second test dataset,
named test private.csv has features of 5000 patients without labels. This dataset will be accessible closer to
the final submission deadline and will be used to compute the final leaderboard of the challenge.
Note that the datasets we provide you have been only partially cleaned. Thus, it is important to try
different pre-processing approaches for feature selection, feature encoding, outliers, etc. You are also encouraged
to explore different trustworthy aspects for your ML modeling and deployment, e.g., exploring fairness and
explainability aspects (see point 5 in Section 3 below).
"""

import re

def format_content(content):
    # Remove extra newlines
    content = re.sub(r'\n+', '\n', content)

    # Remove leading and trailing whitespaces
    content = content.strip()

    # Replace bullet points with proper indentation
    content = re.sub(r'\n•', '\n\n•', content)

    # Replace multiple spaces with a single space
    content = re.sub(r' +', ' ', content)

    # Replace multiple dots with a single dot
    content = re.sub(r'\.{2,}', '.', content)

    # Replace multiple dashes with a single dash
    content = re.sub(r'-{2,}', '-', content)

    # Replace multiple underscores with a single underscore
    content = re.sub(r'_{2,}', '_', content)

    return content

print(format_content(string))