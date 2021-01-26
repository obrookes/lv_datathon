# import modules
# from keras.models import Sequential
# from keras.layers import Dense
import innvestigate
import innvestigate.utils
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
import tensorflow as tf
from tensorflow import set_random_seed
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white")
set_random_seed(12)

# create model for use with gridsearch or randomsearch
def create_model():
	# design network
	ncores = 4
	tf.config.threading.set_intra_op_parallelism_threads(ncores)
	tf.config.threading.set_inter_op_parallelism_threads(ncores)
	model = Sequential()

	model.add(Dense(12, input_dim=69, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='relu'))
	model.compile(loss='mae', optimizer='adam')
	return model

def interpret(train_X,train_y,val_X,val_y,n_epoch,n_batch):
	model = create_model()
	model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(val_X, val_y), verbose=0, shuffle=False)

	# lrp
	analyzer = innvestigate.create_analyzer("lrp.z",model)
	analysis = analyzer.analyze(train_X)
	lrp = pd.DataFrame(np.sum(analysis,axis=0))
	print('lrp',lrp)
	lrp_abs = pd.DataFrame(np.sum(abs(analysis),axis=0))

	# gradient
	analyzer = innvestigate.analyzer.gradient_based.Gradient(model, postprocess=None)
	analysis = analyzer.analyze(train_X)
	grad = pd.DataFrame(np.sum(analysis,axis=0))
	return grad

def get_feature_groups(df):
	dict = {}
	for column in df.columns:
		if column not in ['Country', 'State Code', 'State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Gender', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Claim Reason', 'Sales Channel', 'Vehicle Class', 'Vehicle Size']:
			continue
		sub_categories = list(df[column].drop_duplicates())
		dict[column] = sub_categories
	return dict

def tune(train_X,train_y,val_X,val_y):
	# create model
	model = KerasRegressor(build_fn=create_model)
	# print('Params',model.get_params())
	# define the grid search parameters
	batch_size = [50,100,500,1000]
	epochs = [40,60,80,200]
	param_grid = dict(batch_size=batch_size, epochs=epochs)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,verbose=0)
	grid_result = grid.fit(train_X, train_y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

def fit_model(train_X,train_y,val_X,val_y, n_epoch, n_batch):
	model = create_model()
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
	history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(val_X, val_y), verbose=0, shuffle=False, callbacks=[es])
	# plot history
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.plot(history.history['loss'], label='train',color='#49b6ff')
	plt.plot(history.history['val_loss'], label='test',color='#ff499e')
	plt.legend(frameon=False,prop={'size': 14}) 
	plt.savefig('figs/loss.png',bbox_inches='tight')
	plt.clf()
	model.save('models/NN')
	return model

def evaluate_prediction(pred,actual):
	mae = mean_absolute_error(actual, pred)
	print('t+%d RMSE: ', mae)
	return(rmse)
