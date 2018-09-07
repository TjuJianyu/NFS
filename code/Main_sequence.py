import logging
from Controller import Controller, Controller_sequence, Controller_pure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import argparse
from multiprocessing import Pool, cpu_count, Process
import multiprocessing
from utils import mod_column, evaluate, init_name_and_log, save_result
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
from Java_service import start_service_pool, stop_service_pool, find_free_port
import rpyc
import random
import tensorflow as tf



def parse_args():
	parser = argparse.ArgumentParser(description="Run.")
	parser.add_argument('--num_op_unary', type=int,
		default=4, help='unary operation num')
	parser.add_argument('--num_op_binary', type=int,
		default=5, help='binary operation num')
	parser.add_argument('--max_order', type=int,
		default=5, help='max order of feature operation')
	parser.add_argument('--num_batch', type=int,
		default=32, help='batch num')
	parser.add_argument('--optimizer', nargs='?',
		default='adam', help='choose an optimizer')
	parser.add_argument('--lr', type=float,
		default=0.01, help='set learning rate')
	parser.add_argument('--epochs', type=int,
		default=10000, help='training epochs')
	parser.add_argument('--evaluate', nargs='?',
		default='1-rae', help='choose evaluation method')
	parser.add_argument('--task', nargs='?',
		default='regression', help='choose between classification and regression')
	parser.add_argument('--dataset', nargs='?',
		default='BMI', help='choose dataset to run')
	parser.add_argument('--model', nargs='?',
		default='RF', help='choose a model')
	parser.add_argument('--alpha', type=float,
		default=0.99, help='set discouont factor')
	parser.add_argument('--lr_value', type=float,
		default=1e-3, help='value network learning rate')
	parser.add_argument('--RL_model', nargs='?',
		default='PG', help='choose RL model, PG or AC')
	parser.add_argument('--reg', type=float,
		default=1e-5, help='regularization')
	parser.add_argument('--controller', nargs='?',
		default='rnn', help='choose a controller')
	parser.add_argument('--num_random_sample', type=int,
		default=5, help='sample num of random beseline')
	parser.add_argument('--lambd', type=float,
		default=0.4, help='TD lambd')
	parser.add_argument('--multiprocessing', type=bool,
		default=True, help='whether get reward using multiprocess')
	parser.add_argument('--package', nargs='?',
		default='weka', help='choose sklearn or weka to evaluate')
	return parser.parse_args()


def get_reword(actions):
	global path, args, method, origin_result
	X = pd.read_csv(path)
	num_feature = X.shape[1] - 1
	action_per_feature = int(len(actions) / num_feature)
	copies, copies_run, rewards = {}, [], []

	for feature_count in range(num_feature):
		feature_name = X.columns[feature_count]
		feature_actions = actions[feature_count*action_per_feature: \
			(feature_count+1)*action_per_feature]
		copies[feature_count] = []
		if feature_actions[0] == 0:
			continue
		else:
			copy = np.array(X[feature_name].values)		

		for action in feature_actions:
			if action == 0:
				break
			elif action > 0 and action <= args.num_op_unary:
				action_unary = action - 1
				if action_unary == 0:
					copy = np.squeeze(np.sqrt(abs(copy)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					copy = np.squeeze(scaler.fit_transform(np.reshape(copy,[-1,1])))
				elif action_unary == 2:
					while (np.any(copy == 0)):
						copy = copy + 1e-5
					copy = np.squeeze(np.log(abs(np.array(copy))))
				elif action_unary == 3:
					while (np.any(copy == 0)):
						copy = copy + 1e-5
					copy = np.squeeze(1 / (np.array(copy))) 
			
			else:
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)

				if rank >= feature_count:
					rank += 1
				target_feature_name = X.columns[rank]

				target = np.array(X[target_feature_name].values) 

				if action_binary == 0:
					copy = np.squeeze(copy + target)
				elif action_binary == 1:
					copy = np.squeeze(copy - target)
				elif action_binary == 2:
					copy = np.squeeze(copy * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					copy = np.squeeze(copy / target) 
				elif action_binary == 4:
					copy = np.squeeze(mod_column(copy, X[target_feature_name].values))

			copies[feature_count].append(copy)
		copies_run.append(copy)

	if method == 'train':
		former_result = origin_result
		former_copys = [None]
		for key in sorted(copies.keys()):
			reward, former_result, return_copy = get_reward_per_feature( 
				copies[key], action_per_feature, former_result, former_copys)
			former_copys.append(return_copy)
			rewards += reward
		return rewards
	
	elif method == 'test':
		for i in range(len(copies_run)):
			X.insert(0, 'new%d'%i, copies_run[i])
		if args.package == 'weka':
			result = get_weka_result(X)
		elif args.package == 'sklearn':
			y = X[X.columns[-1]]
			del X[X.columns[-1]]
			result = evaluate(X, y, args)
		return result
		

def get_reward_per_feature(copies, count, former_result, former_copys=[None]):
	global path, args
	
	X = pd.read_csv(path)
	if args.package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]

	reward = []
	previous_result = former_result
	for i,former_copy in enumerate(former_copys):
		if not former_copy is None:
			X.insert(0, 'former%d'%i, former_copy)

	for copy in copies:
		X.insert(0, 'new', copy)
		if args.package == 'weka':
			current_result = get_weka_result(X)
		elif args.package == 'sklearn':
			current_result = evaluate(X, y, args)

		reward.append(current_result - previous_result)
		previous_result = current_result
		del X['new']

	reward_till_now = len(reward)
	for _ in range(count - reward_till_now):
		reward.append(0)
	if len(copies) == 0:
		return_copy = None
	else:
		return_copy = copies[-1]

	return reward, previous_result, return_copy

def random_run(num_random_sample, model, l=None, p=None):
	global args, num_process
	samples = []
	for i in range(num_random_sample):
		sample = []
		for _ in range(model.num_action):
			sample.append(np.random.randint(model.num_op))
		samples.append(sample)

	if args.multiprocessing:	
		if args.package == 'weka':
			pool = Pool(num_process, initializer=init, initargs=(l, p))
		elif args.package == 'sklearn':
			pool = Pool(num_process)    
		res = list(pool.map(get_reword, samples))
		pool.close()
		pool.join()
	else:
		res = []
		for sample in samples:
			res.append(get_reword(sample))

	random_result = max(res)
	random_sample = samples[res.index(random_result)]

	return random_result, random_sample


def train(model, l=None, p=None):
	global path, args, infos, method, origin_result, num_process

	X = pd.read_csv(path)
	if args.package == 'weka':
		origin_result = get_weka_result(X)
	elif args.package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]
		print(X.shape)
		origin_result = evaluate(X, y, args)	
	best_result = origin_result
	print(origin_result)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		init_op = tf.group(tf.global_variables_initializer(), 
			tf.local_variables_initializer())
		sess.run(init_op)
		
		model_result = -10000.0 
		train_set, values = [], []
		for epoch_count in range(args.epochs):
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))

			for batch_count in range(args.num_batch):
				batch_action = []
				for i in range(probs_action.shape[0]):
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])
					batch_action.append(sample_action)
				concat_action.append(batch_action)
					
			method = 'train'
			if args.multiprocessing:
				if args.package == 'weka':
					pool = Pool(num_process, initializer=init, initargs=(l, p))  
				elif args.package == 'sklearn':
					pool = Pool(num_process)        
				rewards = np.array(pool.map(get_reword, concat_action))
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:
					rewards.append(get_reword(action))
				rewards = np.array(rewards)

			method = 'test'
			if args.multiprocessing:
				if args.package == 'weka':
					pool = Pool(num_process, initializer=init, initargs=(l, p))  
				elif args.package == 'sklearn':
					pool = Pool(num_process)      
				results = pool.map(get_reword, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword(action))
			model_result = max(model_result, max(results))


			if args.RL_model == 'AC':
				target_set = []
				for batch_count in range(args.num_batch):
					action = concat_action[batch_count]
					for i in range(model.num_action):
						train_tmp = list(np.zeros(model.num_action, dtype=int))
						target_tmp = list(np.zeros(model.num_action, dtype=int))
						
						train_tmp[0:i] = list(action[0:i])
						target_tmp[0:i+1] = list(action[0:i+1])

						train_set.append(train_tmp)
						target_set.append(target_tmp)

				state = np.reshape(train_set, [-1,model.num_action])
				next_state = np.reshape(target_set, [-1,model.num_action])

				value = model.predict_value(next_state) * args.alpha + rewards.flatten()
				values += list(value)
				model.update_value(state, values)

				rewards_predict = model.predict_value(next_state) * args.alpha - \
					model.predict_value(state[-np.shape(next_state)[0]:]) + rewards.flatten()
				rewards = np.reshape(rewards_predict, [args.num_batch,-1])

			
			elif args.RL_model == 'PG':
				for i in range(model.num_action):
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(args.alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						base_order = base_order * np.power(args.lambd, j) 
						rewards_order = rewards_order.astype(float) 
						rewards_order += base_order.astype(float) 
					rewards[:,i] = (1-args.lambd) * rewards_order
				

			feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), \
				model.rewards: np.reshape(rewards,[args.num_batch,-1])}
			loss_epoch = model.update_policy(feed_dict, sess)


			method = 'test'
			probs_action = sess.run(tf.nn.softmax(model.concat_output))
			best_action = probs_action.argmax(axis=1)
			model_result = max(model_result, get_reword(best_action))

			random_result, random_sample = random_run(args.num_random_sample, model, l, p)

			best_result = max(best_result, model_result)

			print('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f, random_result = %.4f, random_sample = %s' 
				% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result, random_result, str(random_sample)))
			logging.info('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f, random_result = %.4f, random_sample = %s' 
				% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result, random_result, str(random_sample)))

			info = [epoch_count, loss_epoch, origin_result, model_result, random_result]
			infos.append(info)
		
def get_port():
	global lock, ports
	with lock:
		avail_port = [port for port in ports.keys() if ports[port]]
		port = random.choice(avail_port)
		ports[port] = False
	return port

def return_port(port):
	global lock, ports
	with lock:
		ports[port] = True

def init(l, p):
	global lock, ports
	lock = l
	ports = p

def get_weka_result(X):
	d = X.shape[0]
	X_tmp = str(X.values.tolist())
	port = get_port()
	conn = conns[port]
	result = conn.root.evaluate(X_tmp, d, args.task, args.model, args.evaluate)
	return_port(port)
	return result


if __name__ == '__main__':
	args = parse_args()
	origin_result, method, name = None, None, None
	num_process, infos = 64, []
	num_weka_process = num_process
	name = init_name_and_log(args)
	print(name)

	if args.package == 'weka':
		ports = [find_free_port() for _ in range(num_weka_process)]
		port_state = [True for _ in range(num_weka_process)]
		pids = start_service_pool(ports)
		sleep(5)
		conns = {}
		for port in ports:
			conns[port] = rpyc.connect('localhost', port)
		m = multiprocessing.Manager()
		lock = m.Lock()
		ports = m.dict(zip(ports, port_state))

	path = args.dataset + '.csv'
	
	num_feature = pd.read_csv(path).shape[1] - 1
	if args.controller == 'rnn':
		controller = Controller(args, num_feature)
	elif args.controller == 'pure':
		controller = Controller_pure(args, num_feature)
	controller.build_graph()

	if args.package == 'weka':
		train(controller, lock, ports)
	elif args.package == 'sklearn':
		train(controller)

	save_result(infos, name)
	
	if args.package == 'weka':
		stop_service_pool(pids)
		for port in conns.keys():
			conns[port].close()
