from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import svm
import numpy as np
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter
import weka.core.converters as converters
import pandas as pd
import weka.core.jvm as jvm
import sys
from rpyc import Service
from rpyc.utils.server import Server
import os
import logging



def mod_column(c1, c2):
	r = []
	for i in range(c2.shape[0]):
		if c2[i] == 0:
			r.append(0)
		else:
			r.append(np.mod(c1[i],c2[i]))
	return r


def evaluate(X, y, args):
	if args.task == 'regression':
		if args.model == 'LR':
			model = Lasso()
		elif args.model == 'RF':
			model = RandomForestRegressor(n_estimators=10, random_state=0)
		if args.evaluate == 'mae':
			r_mae = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_absolute_error').mean()
			return r_mae
		elif args.evaluate == 'mse':
			r_mse = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_squared_error').mean()
			return r_mse
		elif args.evaluate == 'r2':
			r_r2 = cross_val_score(model, X, y, cv=5).mean()
			return r_r2

	elif args.task == 'classification':
		le = LabelEncoder()
		y = le.fit_transform(y)

		if args.model == 'RF':
			model = RandomForestClassifier(n_estimators=10, random_state=0)
		elif args.model == 'LR':
			model = LogisticRegression(multi_class='ovr')
		elif args.model == 'SVM':
			model = svm.SVC()
		if args.evaluate == 'f_score':
			s = cross_val_score(model, X, y, scoring='f1', cv=5).mean()
		elif args.evaluate == 'auc':
			model = RandomForestClassifier(max_depth=10, random_state=0)
			split_pos = X.shape[0] // 10
			X_train, X_test = X[:9*split_pos], X[9*split_pos:]
			y_train, y_test = y[:9*split_pos], y[9*split_pos:]
			model.fit(X_train, y_train)
			y_pred = model.predict_proba(X_test)
			s = evaluate_(y_test, y_pred)
		return s

def evaluate_(y_true, y_pred):
	num_class = max(y_true) + 1
	y_true = np.eye(num_class)[y_true]
	return 2 * roc_auc_score(y_true, y_pred) - 1

class EvaluationService(Service):
	def exposed_evaluate(self, X, d, task, i_model, i_evl):
		data = np.reshape(eval(X), [d,-1], order='C')
		if task == 'regression':
			if i_model == 'LR':
				data = converters.ndarray_to_instances(
					data, relation='tmp')
				data.class_is_last()
				model = Classifier(classname='weka.classifiers.functions.LinearRegression')
				evl = Evaluation(data)
				evl.crossvalidate_model(model, data, 5, Random(0))
			elif i_model == 'RF':
				data = converters.ndarray_to_instances(
					data, relation='tmp')
				data.class_is_last()
				model = Classifier(classname='weka.classifiers.trees.RandomForest')
				evl = Evaluation(data)
				evl.crossvalidate_model(model, data, 5, Random(0))
			if i_evl == 'mae':
				r_mae = evl.mean_absolute_error
				return r_mae
			elif i_evl == 'mse':
				r_mae = evl.mean_square_error
				return r_mse
			elif i_evl == '1-rae':
				r_one_minus_rae = 1 - evl.relative_absolute_error / 100
				del evl, model, data
				return r_one_minus_rae

		elif task == 'classification':
			le = LabelEncoder()
			data[:,-1] = le.fit_transform(data[:,-1])
			if i_model == 'RF':
				dataRaw = converters.ndarray_to_instances(
					data, relation='tmp')
				weka_filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
				weka_filter.inputformat(dataRaw)
				data = weka_filter.filter(dataRaw)
				data.class_is_last()
				model = Classifier(classname='weka.classifiers.trees.RandomForest')
				evl = Evaluation(data)
				evl.crossvalidate_model(model, data, 5, Random(0))
			elif i_model == 'LR':
				model = LogisticRegression(multi_class='ovr')
			elif i_model == 'SVM':
				model = svm.SVC()
			if i_evl == 'f_score':
				fscore = evl.weighted_f_measure
				del evl, model, data, dataRaw
				if not (fscore >= 0.01 and fscore < 1.01): 
					fscore = 0.01
				return fscore


def init_name_and_log(args):
	name = args.dataset + '_' + args.controller + '_' + args.RL_model + '_' + \
		args.model + '_' + args.package + '_' + args.evaluate + '_' + \
		str(args.num_batch) + '_' + str(args.num_random_sample) + '_' + \
		str(args.max_order) + '_' + args.optimizer + str(args.lr) + '_' + \
		str(args.lr_value) + '_' + str(args.alpha) + '_' + str(args.lambd)

	if not os.path.exists('log'):
		os.mkdir('log')
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename='log/' + name + '.log',
		level=logging.INFO)
	logging.info('--start--')
	return name

def save_result(infos, name):
	if not os.path.exists('result'):
		os.mkdir('result')
	save_path = 'result/' + name + '.txt'

	with open(save_path, 'w') as f:
		for info in infos:
			f.write(str(info) + '\n')
	print(name, 'saved')

class ServerImpl(Server):
    def _accept_method(self, sock):
        try:
            self._authenticate_and_serve_client(sock)
        finally:
            pass


if __name__ == '__main__':

	jvm.start(max_heap_size='5g')
	try:
		port = int(sys.argv[1])
		ServerImpl(EvaluationService, port=port).start()
	finally:
		jvm.stop()
	
	


