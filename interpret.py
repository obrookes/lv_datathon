import pandas as pd
import numpy as np
from utils import interpret,get_feature_groups,create_model,tune,fit_model,evaluate_prediction
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="white")

# get training data
filepath = 'data/new_train.csv'
df = pd.read_csv(filepath)
cat_features = ['Country', 'State Code', 'State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Gender', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Claim Reason', 'Sales Channel', 'Vehicle Class', 'Vehicle Size']
df = pd.get_dummies(data=df,columns=cat_features)
train_y = df['Total Claim Amount']
train_X = df.drop(columns=['Total Claim Amount','Customer','Effective To Date'])

# get validation data
filepath = 'data/val.csv'
df = pd.read_csv(filepath)
df = pd.get_dummies(data=df,columns=cat_features)
val_y = df['Total Claim Amount']
val_X = df.drop(columns=['Total Claim Amount','Customer','Effective To Date'])
columns = val_X.columns

# calculate gradient
tune(train_X,train_y,val_X,val_y)
model = create_model()
print('fitting model')
fit_model(train_X,train_y,val_X,val_y, 100, 200)
grad = interpret(train_X,train_y,val_X,val_y,100,200)
pred = model.predict(val_X, batch_size=200)
evaluate_prediction(pred,val_y)


# plot
x = columns
x_cropped = x.str[:7]
x_cropped = []
for string in x:
	if '_' in string:
		string = string.split('_')[0]
	x_cropped.append(string)
y = [item for sublist in grad.values for item in sublist]
xy = pd.DataFrame({'feature':x,'gradient':y,'feature_cropped':x_cropped}).sort_values(['feature_cropped','gradient'])
xy.to_csv('data/grad.csv')
x = xy['feature']
y = xy['gradient']
colours = []
groups = []
cat_colours = ['#54478c','#2c699a','#048ba8','#0db39e','#16db93','#83e377','#b9e769','#efea5a','#f1c453','#f29e4c','#38D6FA','#9C92C8','#80B2DB','#ff9b85','#ee6055','#c08497']

for feature in x:
	n_colours = len(colours)
	for i,cat_feature in enumerate(cat_features):
		cat_feature_ = '%s_' % cat_feature
		if cat_feature_ in feature:
			colour = cat_colours[i]
			colours.append(colour)
			groups.append(cat_feature)
	n_colours_new = len(colours)	
	if n_colours_new == n_colours:
		colours.append('#ccd7e4')
		groups.append('Continuous')


plt.figure(figsize=(20,5))
plt.bar(x,y,color=colours)
plt.ylabel('Gradient')
plt.title('Neural Network Feature Gradients')
plt.xticks(rotation='vertical')
plt.savefig('figs/nn_interpret.png',bbox_inches='tight')


