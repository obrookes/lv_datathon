
install_a_package = function(package){
  is_installed = library(package, logical.return = TRUE, character.only = TRUE)
  if(is_installed){
    require(package, character.only = TRUE)
  } else if (!is_installed) {
    install.packages(package, repos='https://www.stats.bris.ac.uk/R/')
    require(package, character.only = TRUE)
  }
}

# Libraries
install_a_package("readr")
install_a_package("mgcv")

# Load data etc
source("R/load_data.R")

gf_mean_f = formula(Total_Claim_Amount ~
                      Income +
                      s(Monthly_Premium_Auto, by = Location_Code, k = 20) +
                      s(Monthly_Premium_Auto, by = EmploymentStatus, k = 20) +
                      Location_Code + Vehicle_Class + EmploymentStatus + Coverage +
                      Vehicle_Size + Policy + Education
)
gf_var_f = formula(~s(Monthly_Premium_Auto, by = Location_Code) + Income + Location_Code)

t = system.time(
  gf <- gam(list(gf_mean_f, gf_var_f), data = dat, family = gammals)
)

# predict
train_pred = predict(gf, newdata=dat, type = "response")[,1]
val_pred   = predict(gf, newdata=dat_val, type = "response")[,1]

MAE_train = mean(abs(unlist(y) - train_pred))
MAE_val   = mean(abs(unlist(y_val) - val_pred))

# Print errors
cat("\nGAM errors:\nMAE train: ", MAE_train, "\nMAE val: ", MAE_val, "\n\n")

cat("\nTime Taken:\n", t[3]/60, " minutes \n\n")

write.csv(train_pred, file = "gam_preds_train.csv")
write.csv(val_pred, file = "gam_preds_val.csv")

cat("\n\nOutput saved.\n\n")

