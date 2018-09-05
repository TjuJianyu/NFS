import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, regularizers
import warnings
warnings.filterwarnings("ignore")

class Controller:
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order
		self.reg = args.reg

	
	def _create_rnn(self):
		self.rnns = {}
		for i in range(self.num_feature):
			self.rnns['rnn%d'%i] = tf.contrib.rnn.BasicLSTMCell(
				num_units=self.num_op, name='rnn%d'%i)

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32,
			shape=[self.num_batch,self.num_action], name='concat_action')

		self.rewards = tf.placeholder(tf.float32,
			shape=[self.num_batch,self.num_action], name='rewards')

		self.state = tf.placeholder(tf.int32, 
			shape=[None,self.num_action], name='state')
		self.value = tf.placeholder(tf.float32,
			shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_feature,self.num_op], dtype=np.float32)
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(
			optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):
		self.outputs = {}
		
		for i in range(self.num_feature):
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.float32)
			tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i),
				[1,-1])
			for order in range(self.max_order):
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat(
						[self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)	
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
			loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs)) * self.reg
			loss_reg = 0.0
			for i in range(self.num_feature):
				weights = self.rnns['rnn%d'%i].weights
				for w in weights:
					loss_reg += self.reg * tf.reduce_sum(tf.square(w))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			


	def build_graph(self):
		self._create_rnn()
		self._create_variable()
		self._create_placeholder()
		self._create_inference()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)


class Controller_sequence:
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order

	
	def _create_rnn(self):
		self.rnn = tf.contrib.rnn.BasicLSTMCell(
				num_units=self.num_op, name='rnn')

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32,
			shape=[self.num_batch,self.num_action], name='concat_action')

		self.rewards = tf.placeholder(tf.float32,
			shape=[self.num_batch,self.num_action], name='rewards')

		self.state = tf.placeholder(tf.int32, 
			shape=[None,self.num_action], name='state')
		self.value = tf.placeholder(tf.float32,
			shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = tf.ones(shape=[1, self.num_op], dtype=tf.float32)
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(
			optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):
		self.outputs = {}


		tmp_h = self.rnn.zero_state(1, tf.float32)
		tmp_input = tf.reshape(self.input0, [1,-1])
		for action_count in range(self.num_action):
			tmp_input, tmp_h = self.rnn.__call__(tmp_input, tmp_h)
			if action_count == 0:
				self.concat_output = tmp_input
			else:
				self.concat_output = tf.concat(
					[self.concat_output, tmp_input], axis=0)


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)		
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
			loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs))
			loss_reg = 0.0
			for w in self.rnn.weights:
				loss_reg += 0.01 * tf.nn.l2_loss(w)
	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			


	def build_graph(self):
		self._create_rnn()
		self._create_variable()
		self._create_placeholder()
		self._create_inference()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=25, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)


class Controller_pure:
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order
		self.reg = args.reg



	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32,
			shape=[self.num_batch,self.num_action], name='concat_action')

		self.rewards = tf.placeholder(tf.float32,
			shape=[self.num_batch,self.num_action], name='rewards')

		self.state = tf.placeholder(tf.int32, 
			shape=[None,self.num_action], name='state')
		self.value = tf.placeholder(tf.float32,
			shape=[None,1], name='value')


	def _create_variable(self):

		self.input0 = np.ones(shape=[self.num_action,self.num_op], dtype=np.float32)
		self.input0 = self.input0 / self.num_op
		self.concat_output = tf.Variable(
			initial_value=self.input0, name='concat_output', dtype=tf.float32)

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(
			optimizer=self.value_optimizer, loss='mean_squared_error')
		


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)
			
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)

			self.reward_test = reward
			loss_batch = -tf.reduce_sum(reward * tf.log(tf.clip_by_value(pick_action_prob,1e-10,1.0)))
			loss_entropy = -tf.reduce_sum(action_probs * tf.log(tf.clip_by_value(action_probs,1e-10,1.0))) * self.reg

			self.loss += loss_batch + loss_entropy
		self.loss /= self.num_batch



	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			


	def build_graph(self):
		self._create_variable()
		self._create_placeholder()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss, reward_test = sess.run([self.optimizer,self.loss, self.reward_test], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)
