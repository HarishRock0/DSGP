"""
Shared lookup tables for LFS-2023 Sri Lanka dataset.
Imported by both NLPC.py and LLMQ.py to avoid duplication.
"""

COLUMN_DESCRIPTIONS = {
    # Identification & Location
    "YEAR": "Survey year (2023)",
    "MONTH": "Survey month (1-12)",
    "SECTOR": "Residential Sector (1=Urban, 2=Rural, 3=Estate)",
    "DISTRICT": "Administrative District code (11=Colombo, 12=Gampaha, 13=Kalutara, 21=Kandy, 22=Matale, 23=Nuwara Eliya, 31=Galle, 32=Matara, 33=Hambantota, 41=Jaffna, 42=Kilinochchi, 43=Mannar, 44=Vavuniya, 45=Mullaitivu, 51=Batticaloa, 52=Ampara, 53=Trincomalee, 61=Kurunegala, 62=Puttalam, 71=Anuradhapura, 72=Polonnaruwa, 81=Badulla, 82=Monaragala, 91=Ratnapura, 92=Kegalle)",
    "PSU": "Primary Sampling Unit",
    "HUNIT": "Housing Unit number",
    "HHOLD": "Household number",
    "SERNO": "Serial number of person within household",
    # Demographics
    "RSHIP": "Relationship to head of household (1=Head, 2=Spouse, 3=Child, 4=Parent, 5=Other relative, 6=Non-relative)",
    "SEX": "Gender (1=Male, 2=Female)",
    "BYEAR": "Birth year",
    "BMONTH": "Birth month",
    "AGE": "Age in years (numeric)",
    "ETH": "Ethnic Group (1=Sinhala, 2=SL Tamil, 3=Indian Tamil, 4=Moor, 5=Malay, 6=Burgher, 9=Other)",
    "REL": "Religion (1=Buddhist, 2=Hindu, 3=Islam, 4=Roman Catholic, 5=Other Christian, 9=Other)",
    "MARITAL": "Marital Status (1=Never Married, 2=Married, 3=Widowed, 4=Divorced, 5=Separated)",
    "EDU": "Highest Education Level (0/19=No schooling, 1-10=Grade 1-10, 11=O/L, 12=Passed O/L, 13=A/L, 14=Passed A/L, 15=Degree, 16=Postgraduate)",
    "DEGREE": "Degree field of study (if applicable)",
    "CUEDU": "Currently in Education (1=Yes, 2=No)",
    # Literacy
    "SIN": "Sinhala Literacy (1=Can read/write, 2=Cannot read/write)",
    "TAMIL": "Tamil Literacy (1=Can read/write, 2=Cannot read/write)",
    "ENG": "English Literacy (1=Can read/write, 2=Cannot read/write)",
    # Disability (P15-P20 all use: 1=None, 2=Some, 3=A lot, 4=Cannot do)
    "P15": "Vision Difficulty - Even with glasses (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P16": "Hearing Difficulty - Even with hearing aid (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P17": "Mobility/Walking Difficulty - Walking or climbing steps (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P18": "Cognitive Difficulty - Remembering or concentrating (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P19": "Self-care Difficulty - Washing or dressing (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P20": "Communication Difficulty - Using usual language (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P21": "Education/Training Participation - Last 12 months",
    # Employment
    "Q2": "Work Activity - Did person work for pay/profit in last 7 days? (1=Yes, 2=No)",
    "Q8": "Occupation code - Main job/task performed (ISCO-08 coded, stored as string)",
    "Q16": "Employment Status (1=Employee public/private, 2=Employer, 3=Own account worker/self-employed, 4=Contributing family worker)",
    "Q20": "Hours Worked - Total actual hours per week at main job (underemployment if < 40)",
    "Q36": "Job Search - Looked for work in reference period (1=Yes, 2=No)",
    "Q43": "Availability - Available to start work (1=Yes, 2=No)",
    # Income & Poverty
    "Q45_A_1": "Monthly Income/Salary in LKR - stored as STRING with spaces for missing values. Must convert to numeric with pd.to_numeric(errors='coerce'). Only ~2600 of ~18937 rows have actual numeric values.",
    # Formality & Benefits
    "Q46": "EPF/ETF Benefits (1=Yes/formal, 2=No/informal)",
    "Q47": "Workplace Formality (1=Formal/Registered, 2=Informal/Not registered)",
    # Digital Skills
    "Q60A": "Computer Literacy (1=Can use, 2=Cannot use)",
    "Q60B": "Smartphone/Tablet (1=Can use, 2=Cannot use)",
    "Q61": "Internet Use - Used internet in last 12 months (1=Yes, 2=No)",
    "Q64": "Internet Use frequency",
    # Weighting
    "Annual_Factor": "Survey weight / Annual expansion factor for population estimates",
}

COLUMN_VALUE_SCALE = {
    1: "No difficulty/None",
    2: "Some difficulty/Minor",
    3: "A lot of difficulty/Major",
    4: "Cannot do at all/Severe",
}

EMPLOYMENT_STATUS = {
    1: "Employee (public/private)",
    2: "Employer",
    3: "Own Account Worker (Self-employed)",
    4: "Contributing Family Worker",
}

SECTOR_MAP = {
    1: "Urban",
    2: "Rural",
    3: "Estate",
}

DISTRICT_MAP = {
    11: "Colombo",    12: "Gampaha",    13: "Kalutara",
    21: "Kandy",      22: "Matale",     23: "Nuwara Eliya",
    31: "Galle",      32: "Matara",     33: "Hambantota",
    41: "Jaffna",     42: "Kilinochchi",43: "Mannar",
    44: "Vavuniya",   45: "Mullaitivu",
    51: "Batticaloa", 52: "Ampara",     53: "Trincomalee",
    61: "Kurunegala", 62: "Puttalam",
    71: "Anuradhapura", 72: "Polonnaruwa",
    81: "Badulla",    82: "Monaragala",
    91: "Ratnapura",  92: "Kegalle",
}

ETHNICITY_MAP = {
    1: "Sinhala", 2: "SL Tamil", 3: "Indian Tamil",
    4: "Moor", 5: "Malay", 6: "Burgher", 9: "Other",
}

RELIGION_MAP = {
    1: "Buddhist", 2: "Hindu", 3: "Islam",
    4: "Roman Catholic", 5: "Other Christian", 9: "Other",
}

MARITAL_MAP = {
    1: "Never married", 2: "Married", 3: "Widowed",
    4: "Divorced", 5: "Separated",
}

PROVINCE_DISTRICTS = {
    "Western":       [11, 12, 13],
    "Central":       [21, 22, 23],
    "Southern":      [31, 32, 33],
    "Northern":      [41, 42, 43, 44, 45],
    "Eastern":       [51, 52, 53],
    "North Western": [61, 62],
    "North Central": [71, 72],
    "Uva":           [81, 82],
    "Sabaragamuwa":  [91, 92],
}
