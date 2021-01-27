from numpy.random import seed
seed(1)
import random
random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import numpy as np
from utils import interpret,get_feature_groups,create_model,tune,fit_model,evaluate_prediction
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tensorflow as tf
#6 Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
sns.set(style="white")


# get training data
filepath = 'data/new_train.csv'
df = pd.read_csv(filepath)
cat_features = ['Country', 'State Code', 'State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Gender', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Claim Reason', 'Sales Channel', 'Vehicle Class', 'Vehicle Size']
df = pd.get_dummies(data=df,columns=cat_features).drop(columns=['Customer','Effective To Date'])
# scale and transform data
train_y = df['Total Claim Amount']
train_X = df.drop(columns=['Total Claim Amount'])
columns = train_X.columns
scalerX = MinMaxScaler().fit(np.array(train_X))
scalery = MinMaxScaler().fit(np.array(train_y).reshape(-1,1))
train_X = pd.DataFrame(scalerX.transform(np.array(train_X)))
train_y = pd.DataFrame(scalery.transform(np.array(train_y).reshape(-1,1)))
train_X.columns = columns

# get validation data
filepath = 'data/val.csv'
df = pd.read_csv(filepath)
df = pd.get_dummies(data=df,columns=cat_features).drop(columns=['Customer','Effective To Date'])
val_y = df['Total Claim Amount']
val_X = df.drop(columns=['Total Claim Amount'])
columns = val_X.columns
val_X = pd.DataFrame(scalerX.transform(np.array(val_X)))
val_y = pd.DataFrame(scalery.transform(np.array(val_y).reshape(-1,1)))
val_X.columns = columns

# calculate gradient
n_epoch,n_batch = tune(train_X,train_y,val_X,val_y)
print(n_epoch)
print(n_batch)
model = create_model()
print('fitting model')

fit_model(train_X,train_y,val_X,val_y, n_epoch, n_batch)
grad = interpret(train_X,train_y,val_X,val_y,n_epoch,n_batch)
pred = model.predict(val_X, batch_size=n_batch)
mae = evaluate_prediction(pred,val_y)
print(pred)
print(val_y)

pred = scalery.inverse_transform(pred)
val_y = scalery.inverse_transform(val_y)
print(pred)
print(val_y)
plt.figure(figsize=(20,5))
plt.scatter(range(len(pred)),pred)
plt.scatter(range(len(val_y)),val_y)
plt.savefig('figs/predictions.png')
plt.clf()

# print(val_y)
mae = evaluate_prediction(pred,val_y)

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


