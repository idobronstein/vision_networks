import os
import time
import sys
import shutil
from datetime import timedelta
from functools import reduce

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

K_MEANS_ITERATION_NUM = 20

class CompreseDenseNet:
	def __init__(self, densenet_model, new_growth_rate):
		self.densenet_model = densenet_model
		self.new_growth_rate = new_growth_rate
		self.k_means =  KMeans(n_clusters=self.new_growth_rate, algorithm="full", random_state=0)
		self.graph = tf.get_default_graph()
		self.bottleneck_output_size = self.densenet_model.growth_rate*4

	def gather_all_kernels_block(self, kernel_type, block_num):
		all_kernel = []
		for layer_num in range(self.densenet_model.layers_per_block):
			name = 'Block_{block_num}/layer_{layer_num}/{kernel_type}/kernel:0'.format(
					block_num=block_num, layer_num=layer_num, kernel_type=kernel_type)
			kernel_tensor = self.graph.get_tensor_by_name(name)
			all_kernel.append(kernel_tensor)
		return all_kernel

	def gather_all_batch_norm_block(self, kernel_type, block_num):
		all_batch_norm = []
		names = ['Block_{block_num}/layer_{layer_num}/{kernel_type}/BatchNorm/beta:0',
			'Block_{block_num}/layer_{layer_num}/{kernel_type}/BatchNorm/gamma:0',
			'Block_{block_num}/layer_{layer_num}/{kernel_type}/BatchNorm/moving_mean:0',
			'Block_{block_num}/layer_{layer_num}/{kernel_type}/BatchNorm/moving_variance:0']
		for layer_num in range(self.densenet_model.layers_per_block):
			batch_norm = []
			for name in names:
				name = name.format(block_num=block_num, layer_num=layer_num, kernel_type=kernel_type)
				kernel_tensor = self.graph.get_tensor_by_name(name)
				batch_norm.append(kernel_tensor)
			all_batch_norm.append(batch_norm)
		return all_batch_norm

	def get_transition_kernel(self, block_num):
		name = 'Transition_after_block_{block_num}/composite_function/kernel:0'.format(block_num=block_num)
		kernel_tensor = self.graph.get_tensor_by_name(name)
		return kernel_tensor

	def get_transition_batch_norm(self, block_num):
		names = ['Transition_after_block_{block_num}/composite_function/BatchNorm/beta:0',
				 'Transition_after_block_{block_num}/composite_function/BatchNorm/gamma:0',
				 'Transition_after_block_{block_num}/composite_function/BatchNorm/moving_mean:0',
				 'Transition_after_block_{block_num}/composite_function/BatchNorm/moving_variance:0']
		batch_norm = []
		for name in names:
			name = name.format(block_num=block_num)
			kernel_tensor = self.graph.get_tensor_by_name(name)
			batch_norm.append(kernel_tensor)
		return batch_norm

	def get_tansition_to_classes_batch_norm(self):
		names = ['Transition_to_classes/BatchNorm/beta:0',
				 'Transition_to_classes/BatchNorm/gamma:0',
				 'Transition_to_classes/BatchNorm/moving_mean:0',
				 'Transition_to_classes/BatchNorm/moving_variance:0']
		batch_norm = []
		for name in names:
			kernel_tensor = self.graph.get_tensor_by_name(name)
			batch_norm.append(kernel_tensor)
		return batch_norm

	def kernel_to_cluster_centers(self, cluster_centers, cluster_indices):
		h, w, i = cluster_centers[0].shape
		new_kernel_vector = np.n([h, w, i, self.densenet_model.growth_rate])
		for kernel_index, cluster_index  in enumerate(cluster_indices):			
			new_kernel_vector[:,:,:,kernel_index] = cluster_centers[cluster_index]
		delta = abs(composite_layer_vactor - new_kernel_vector).sum()
		print("{name} delta: {delta}".format(name=composite_layer.name, delta=delta))
		return new_kernel_vector

	def cluster_composite_layer(self, composite_layer):
		h, w, i, o = composite_layer.shape
		composite_layer_vactor = self.densenet_model.sess.run(composite_layer)
		composite_layer_vactor_move = np.moveaxis(composite_layer_vactor, -1, 0)
		composite_layer_vactor_reshape = np.reshape(composite_layer_vactor_move, [o, h*w*i])
		k_meas_res = self.k_means.fit(composite_layer_vactor_reshape)
		cluster_indices = k_meas_res.labels_
		cluster_centers_vector = k_meas_res.cluster_centers_
		#cluster_centers = [np.reshape(cluster_centers_vector[k], [h, w, i]) for k in cluster_indices]
		cluster_centers = [np.reshape(cluster_centers_vector[k], [h, w, i]) for k in range(self.new_growth_rate)]
		cluster_centers = np.moveaxis(cluster_centers, 0, 3)
		#cluster_indices = range(12)
		return cluster_centers, cluster_indices

	def create_orignatl_to_comprese_matrix(self, cluster_indices):
		OtC = np.zeros((self.new_growth_rate, self.densenet_model.growth_rate))
		for cluster_num in range(self.new_growth_rate):
			for index in range(self.densenet_model.growth_rate):
				if cluster_num == cluster_indices[index]:
					OtC[cluster_num, index] = 1
		return OtC

	def adjust_bottleneck_kernel(self, bottleneck_kernel, original_to_comprese, layer_num):
		bottleneck_kernel_vector = self.densenet_model.sess.run(bottleneck_kernel)
		
		if layer_num == 0:
			return bottleneck_kernel_vector
		
		for i in range(layer_num):
			start_index = i*self.densenet_model.growth_rate
			bottleneck_kernel_slice = bottleneck_kernel_vector[:, :, start_index : start_index + self.densenet_model.growth_rate, :]
			adjust_bottleneck_slice = np.tensordot(original_to_comprese[(layer_num - 1) - i], bottleneck_kernel_slice, (1, 2))
			adjust_bottleneck_slice = np.moveaxis(adjust_bottleneck_slice, 0, 2)
			if i == 0:
				adjust_bottleneck = adjust_bottleneck_slice
			else:
				adjust_bottleneck = np.concatenate((adjust_bottleneck, adjust_bottleneck_slice), axis=2)

		from_uncahnge_input =  bottleneck_kernel_vector[:, :, start_index + self.densenet_model.growth_rate : , :]
		adjust_bottleneck = np.concatenate((adjust_bottleneck, from_uncahnge_input), axis=2)
		return adjust_bottleneck

	def adjust_batch_norm(self, batch_norm, original_to_comprese, layer_num):
		batch_norm_vector = self.densenet_model.sess.run(batch_norm)
		if layer_num == 0:
			return batch_norm_vector
		new_batch_norm_vector = []
		for param in batch_norm_vector:
			for i in range(layer_num):
				start_index = i*self.densenet_model.growth_rate
				param_slice = param[start_index : start_index + self.densenet_model.growth_rate]
				adjust_param_slice = np.tensordot(original_to_comprese[(layer_num - 1) - i], param_slice, (1, 0))
				if i == 0:
					adjust_param = adjust_param_slice
				else:
					adjust_param = np.concatenate((adjust_param, adjust_param_slice), axis=0)
			from_uncahnge_input =  param[start_index + self.densenet_model.growth_rate :]
			adjust_param = np.concatenate((adjust_param, from_uncahnge_input), axis=0)
			new_batch_norm_vector.append(adjust_param)

		return new_batch_norm_vector

	def adjust_W(self, W, original_to_comprese, layer_num):
		W_vector = self.densenet_model.sess.run(W)
		if layer_num == 0:
			return W_vector
		for i in range(layer_num):
			start_index = i*self.densenet_model.growth_rate
			W_slice = W_vector[start_index : start_index + self.densenet_model.growth_rate, :]
			adjust_W_slice = np.tensordot(original_to_comprese[(layer_num - 1) - i], W_slice, (1, 0))
			if i == 0:
				adjust_W = adjust_W_slice
			else:
				adjust_W = np.concatenate((adjust_W, adjust_W_slice), axis=0)
		from_uncahnge_input =  W_vector[start_index + self.densenet_model.growth_rate :, :]
		adjust_W = np.concatenate((adjust_W, from_uncahnge_input), axis=0)
		return adjust_W

	def comprese(self):
		all_new_comprese_kernels = []
		all_new_bottleneck_kernels = []
		all_new_batch_norm = []
		all_new_transion_kernels = []
		all_new_batch_norm_for_transion = []
		for block_num in range(self.densenet_model.total_blocks): 
			# calc comprese composite kernels
			composite_kernels = self.gather_all_kernels_block('composite_function', block_num)
			block_new_composite_kernels = []
			original_to_comprese = []
			for composite_kernel in composite_kernels:
				cluster_centers, cluster_indices = self.cluster_composite_layer(composite_kernel)
				OtC = self.create_orignatl_to_comprese_matrix(cluster_indices)
				block_new_composite_kernels.append(cluster_centers)				
				original_to_comprese.append(OtC)
			all_new_comprese_kernels.append(block_new_composite_kernels)

			# calc new bottleneck kernels
			bottleneck_kernels = self.gather_all_kernels_block('bottleneck', block_num)
			bottleneck_batch_norm = self.gather_all_batch_norm_block('bottleneck', block_num)
			block_new_bottleneck_kernels = []
			block_new_batch_norm = []
			for i in range(len(bottleneck_kernels)):
				block_new_bottleneck_kernels.append(self.adjust_bottleneck_kernel(bottleneck_kernels[i], original_to_comprese, i))
				block_new_batch_norm.append(self.adjust_batch_norm(bottleneck_batch_norm[i], original_to_comprese, i))
			all_new_bottleneck_kernels.append(block_new_bottleneck_kernels)
			all_new_batch_norm.append(block_new_batch_norm)

			if block_num != self.densenet_model.total_blocks - 1:
				# calc new transition layer
				transion_layer = self.get_transition_kernel(block_num)
				transion_batch_norm = self.get_transition_batch_norm(block_num)
				all_new_transion_kernels.append(self.adjust_bottleneck_kernel(transion_layer, original_to_comprese, len(bottleneck_kernels)))
				all_new_batch_norm_for_transion.append(self.adjust_batch_norm(transion_batch_norm, original_to_comprese, len(bottleneck_kernels)))
			else:
				W = self.graph.get_tensor_by_name('Transition_to_classes/W:0')
				transion_to_class_batch_norm = self.get_tansition_to_classes_batch_norm()
				new_W = self.adjust_W(W, original_to_comprese, len(bottleneck_kernels))
				new_transion_to_class_batch_norm = self.adjust_batch_norm(transion_to_class_batch_norm, original_to_comprese, len(bottleneck_kernels))

		return all_new_comprese_kernels, all_new_bottleneck_kernels, all_new_batch_norm, all_new_transion_kernels, all_new_batch_norm_for_transion, new_W, new_transion_to_class_batch_norm

	def cluster(self):
		all_new_kernels = []
		for block_num in range(self.densenet_model.total_blocks): 
			block_new_kernels = []
			composite_kernels = self.gather_all_kernels_block('composite_function', block_num)
			for composite_kernel in composite_kernels:
				cluster_centers, cluster_indices = self.cluster_composite_layer(composite_kernel)
				new_kernel_vector = self.kernel_to_cluster_centers(cluster_centers, cluster_indices)
				block_new_kernels.append(new_kernel_vector)
			all_new_kernels.append(block_new_kernels)
		return all_new_kernels
