#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Student Name : Luis E. Ariza
# Cohort       : MSBA1 Lombard



################################################################################
# Import Packages
################################################################################

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score 

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

################################################################################
# Load Data
################################################################################

file = "Apprentice_Chef_Dataset.xlsx"

df = pd.read_excel(file)

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

domain_vector = []

for email in df.loc[:, "EMAIL"]:
    
    domain_vector.append(email.split(sep = "@")[1])
    
df.loc[:, "email_domain"] = domain_vector

# Emails are classified according to the information provided by Marketing

professional = ["mmm.com", "amex.com", "apple.com", "boeing.com",
               "caterpillar.com", "chevron.com", "cisco.com",
               "cocacola.com", "disney.com", "dupont.com", "exxon.com",
               "ge.org", "goldmansacs.com", "homedepot.com", "ibm.com",
               "intel.com", "jnj.com", "jpmorgan.com", "mcdonalds.com",
               "merck.com", "microsoft.com", "nike.com", "pfizer.com",
               "pg.com", "travelers.com", "unitedtech.com", "unitedhealth.com",
               "verizon.com", "visa.com", "walmart.com"]

personal = ["gmail.com", "yahoo.com", "protonmail.com"]

junk = ["me.com", "aol.com", "hotmail.com", "live.com", "msn.com", "passport.com"]

email_type_vector = []

for email in df.loc[:, "email_domain"]:
    
    if email in professional:
        
        email_type_vector.append("professional")
        
    elif email in personal:
        
        email_type_vector.append("personal")
        
    elif email in junk:
        
        email_type_vector.append("junk")
        
    else:
        
        email_type_vector.append("other")
    
df.loc[:, "email_type"] = email_type_vector

# Creating dummy variables based on email type

email_dummies = pd.get_dummies(df.loc[:, "email_type"])

df = df.join(email_dummies)

# Dropping email related variables after encoding. 

# EMAIL will be kept for a further step

df = df.drop("email_type", axis = 1)

# Dropping one of the dummy variables created to avoid colinearity

df = df.drop("personal", axis = 1)

# Extra categorization is created

high_loyalty = ["merck.com", "microsoft.com", "pg.com", "jpmorgan.com", "intel.com"]

mid_loyalty = ["amex.com", "caterpillar.com", "unitedtech.com",
               "goldmansacs.com", "unitedhealth.com", "verizon.com",
              "walmart.com", "pfizer.com", "cocacola.com", "mcdonalds.com",
              "boeing.com", "cisco.com", "nike.com", "apple.com", "mmm.com",
              "ge.org", "travelers.com", "visa.com"]

low_loyalty = ["ibm.com", "dupont.com", "chevron.com", "disney.com", 
               "jnj.com", "exxon.com"]

no_loyalty = ["homedepot.com", "passport.com", "me.com", "live.com", 
              "msn.com", "hotmail.com", "aol.com"]

public_domains = ["protonmail.com", "gmail.com", "yahoo.com"]

# New variables are created based on the new categorization

email_type_vector = []

for email in df.loc[:, "email_domain"]:
    
    if email in high_loyalty:
        
        email_type_vector.append("high_loyalty")
        
    elif email in mid_loyalty:
        
        email_type_vector.append("mid_loyalty")
        
    elif email in low_loyalty:
        
        email_type_vector.append("low_loyalty")
        
    elif email in no_loyalty:
        
        email_type_vector.append("no_loyalty")
        
    elif email in public_domains:
        
        email_type_vector.append("public_domains")
        
    else:
        
        email_type_vector.append("other")
    
df.loc[:, "email_type"] = email_type_vector

# Creating dummy variables based on email type

email_dummies = pd.get_dummies(df.loc[:, "email_type"])

df = df.join(email_dummies)

# Dropping email related variables after encoding

df = df.drop("EMAIL", axis = 1)

df = df.drop("email_type", axis = 1)

# Dropping one of the dummy variables created to avoid colinearity

df = df.drop("low_loyalty", axis = 1)

################################################################################
# Train/Test Split
################################################################################

vars_2 = ["FOLLOWED_RECOMMENDATIONS_PCT", "professional", "junk", 
          "high_loyalty", "mid_loyalty", "no_loyalty", "public_domains"]

df_data = df.loc[:, vars_2]

df_target = df.loc[:, "CROSS_SELL_SUCCESS"]

X_train, X_test, y_train, y_test = train_test_split(df_data,
                                                   df_target,
                                                   test_size = 0.25,
                                                   random_state = 222)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

gb = GradientBoostingClassifier(loss = "deviance", n_estimators = 500)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

c_mat = confusion_matrix(y_true = y_test,
                       y_pred = gb_pred)


################################################################################
# Final Model Score (score)
################################################################################

test_score = roc_auc_score(y_true  = y_test,
              y_score = gb_pred)

print("Confusion Matrix :\n", c_mat)

print("Training score :", gb.score(X_train, y_train).round(4))

print("Testing score :", gb.score(X_test, y_test).round(4))

print("Sensitivity :", (c_mat[1][1]/(c_mat[1][1]+c_mat[1][0])).round(4))

print("Specificity :", (c_mat[0][0]/(c_mat[0][0]+c_mat[0][1])).round(4))

print("AUC :", test_score.round(4))

