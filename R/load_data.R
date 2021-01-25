# Libraries
library(readr)

# Read data
X     = read_csv("Rdata/X.csv")
X_val = read_csv("Rdata/X_val.csv")
y     = read_csv("Rdata/y.csv", col_names = FALSE)
y_val = read_csv("Rdata/y_val.csv", col_names = FALSE)

cont_vars = c("Effective To Date", "Income", "Monthly Premium Auto",
              "Months Since Last Claim", "Months Since Policy Inception",
              "Number of Open Complaints", "Number of Policies")

cat_vars = c("State", "Response", "Coverage", "Education", "EmploymentStatus",
             "Gender", "Location Code", "Marital Status", "Policy Type",
             "Policy", "Claim Reason", "Sales Channel", "Vehicle Class",
             "Vehicle Size")

cont_vars = stringr::str_replace_all(cont_vars, " ", "_")
cat_vars  = stringr::str_replace_all(cat_vars, " ", "_")

rename    = stringr::str_replace_all(colnames(X), " ", "_")
colnames(X) = rename
colnames(X_val) = rename

# Make data frame with predictions
dat     = data.frame(X, stringsAsFactors = TRUE)
dat_val = data.frame(X_val, stringsAsFactors = TRUE)
dat$Total_Claim_Amount = unlist(y)

# Convert strings to factors
for(var in cat_vars){
  dat[, var] = factor(dat[, var])
}