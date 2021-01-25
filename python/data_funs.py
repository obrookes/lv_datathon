import pandas as pd
import numpy as np
from dateutil import parser

def set_up(dat):
        
    # Remove customer ID, all countries are US, state is repeated
    dat = dat.drop(columns = ["Customer", "Country", "State Code"])
    
    # Convert 'Effective To Date' to continuous
    def convert_dates(x):
        return (parser.parse(x) - parser.parse("01/01/2011")).days
    dat["Effective To Date"] = dat["Effective To Date"].apply(convert_dates)
    
    # Set up continuous/categorical columns
    cont = ["Income", "Monthly Premium Auto", "Months Since Last Claim", 
            "Months Since Policy Inception", "Number of Open Complaints",
            "Number of Policies", "Effective To Date"]
    cat  = list(dat.columns.drop(cont).drop("Total Claim Amount"))
    
    # Subset to just categorical
    y = dat.loc[:, "Total Claim Amount"]
    X = dat.drop(columns = "Total Claim Amount")
    
    # Convert to one-hot encoded vectors
    for i in cat:
        category = X.loc[:, i].unique()
        X.loc[:, i] = pd.Categorical(X.loc[:, i], categories = category)
    
    return X, y

def onehot(X):
    
    # Get columns that are categories
    cat = X.columns[X.dtypes == "category"]
    
    # One hot those, remove them and add the one-hotted back in
    X_onehot = pd.get_dummies(X.loc[:, cat])
    X = X.drop(columns = cat)
    X = X.join(X_onehot)
    
    return X

def subset(X, var = "both"):
    cont = ["Monthly Premium Auto", "Income", "Number of Open Complaints"]
    cat  = ["Location Code", "Vehicle Class", "EmploymentStatus", 
            "Coverage", "Education", "Policy", "Vehicle Size"]
    both = np.hstack((cont, cat))
    
    Y = X.copy()
    if var == "both":
        Y = Y.loc[:, both]
    elif var == "cont":
        Y = Y.loc[:, cont]
    elif var == "cat":
        Y = Y.loc[:, cat]
            
    return Y