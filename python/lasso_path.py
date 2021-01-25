# Use LASSO path algorithm to choose which features to include
# Files
import python.data_funs as df

# Libraries
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv("data/train.csv")
X, y = df.set_up(train)
X = df.onehot(X)

# LASSO 
X /= X.std(axis=0)
alphas_lasso, coefs_lasso, coef = lm.lasso_path(X, y, eps= 5e-3, fit_intercept=False)

# Plot path
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for i in range(coefs_lasso.shape[0]):
    plt.plot(neg_log_alphas_lasso, coefs_lasso[i, :], label=X.columns[i])

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso Paths')
plt.legend(loc='lower left')
plt.axis('tight')
plt.show()

# LASSO CV
reg = lm.LassoCV(max_iter = 50000, verbose = 0, random_state = 1, tol = 1e-5)
reg.fit(X, y)
coef = pd.Series(reg.coef_, index = X.columns)
imp_coef = coef.sort_values()
imp_coef.plot(kind = "barh")
plt.title("Feature importance from LASSO")
plt.show()

# Filter coefficients that reduced to zero during LASSO (irrelevant)
largest_coefs = coef[abs(coef) - 1e-1 >= 0]
largest_coefs = largest_coefs.index.str.split('_').str[0].unique()

# More harsh filtering of coefficients
harsh_coefs = coef[abs(coef) - 2 >= 0]
harsh_coefs = harsh_coefs.index.str.split('_').str[0].unique()


# Save coefficients and raw values
pd.Series(largest_coefs).to_csv("model/coef_lasso.csv", index = None, header = False)
coef.to_csv("model/coef_lasso_raw.csv", header = False)


































