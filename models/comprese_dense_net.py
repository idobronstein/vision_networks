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
K0 = [24, 108, 150]

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

	def get_transition_kernel(self, block_num):
		name = 'Transition_after_block_{block_num}/composite_function/kernel:0'.format(block_num=block_num)
		kernel_tensor = self.graph.get_tensor_by_name(name)
		return kernel_tensor

	def cluster_composite_layer(self, composite_layer):
		h, w, i, o = composite_layer.shape
		composite_layer = tf.reshape(composite_layer, [o, h*w*i])
		composite_layer_vactor = self.densenet_model.sess.run(composite_layer)
		k_meas_res = self.k_means.fit(composite_layer_vactor)
		cluster_indices = k_meas_res.labels_
		cluster_centers_vector = k_meas_res.cluster_centers_
		#cluster_centers = tf.convert_to_tensor(cluster_centers_vector, dtype=tf.float32)
		#cluster_centers = tf.reshape(cluster_centers, [h, w, i, self.new_growth_rate])
		cluster_centers = np.reshape(cluster_centers_vector, [h, w, i, self.new_growth_rate])
		return cluster_centers, cluster_indices

	def create_orignatl_to_comprese_matrix(self, cluster_indices):
		assert len(cluster_indices) == self.densenet_model.growth_rate
		OtC = np.zeros((self.new_growth_rate, self.densenet_model.growth_rate))
		for i in cluster_indices:
			OtC[cluster_indices[i],[i]] = 1
		OtC = tf.convert_to_tensor(OtC, dtype=tf.float32)
		return OtC

	def adjust_bottleneck_kernel(self, bottleneck_kernel, original_to_comprese, k0, layer_num):
		# TODO: reorder the matrix
		output_size = bottleneck_kernel.shape[-1]
		# don't touch k0 first params
		adjust_bottleneck = bottleneck_kernel[:, :, :k0, :]
		if not layer_num == 0:
			# for every input unite his wights by clusters
			for i in range(1, layer_num + 1):
				for j in range(i):
					start_index = k0 + j*self.densenet_model.growth_rate
					bottleneck_kernel_slice = bottleneck_kernel[:, :, start_index : start_index + self.densenet_model.growth_rate, :]
					adjust_bottleneck_slice = tf.tensordot(original_to_comprese[j], bottleneck_kernel_slice, [[1], [2]])
					adjust_bottleneck_slice = tf.reshape(adjust_bottleneck_slice, [1, 1, self.new_growth_rate, output_size])
					adjust_bottleneck = tf.concat(axis=2, values=(adjust_bottleneck, adjust_bottleneck_slice))
		return adjust_bottleneck

	def comprese(self):
		for block_num in range(self.densenet_model.total_blocks): 
			k0 = K0[block_num]
			# calc comprese composite kernels
			composite_kernels = self.gather_all_kernels_block('composite_function', block_num)
			comprese_composite_kernels = []
			original_to_comprese = []
			for composite_kernel in composite_kernels:
				cluster_centers, cluster_indices = self.cluster_composite_layer(composite_kernel)
				OtC = self.create_orignatl_to_comprese_matrix(cluster_indices)
				comprese_composite_kernels.append(cluster_centers)
				original_to_comprese.append(OtC)

			# calc new bottleneck kernels
			bottleneck_kernels = self.gather_all_kernels_block('bottleneck', block_num)
			new_bottleneck_kernels = []
			for i, bottleneck_kernel in enumerate(bottleneck_kernels):
				new_bottleneck_kernel = self.adjust_bottleneck_kernel(bottleneck_kernel, original_to_comprese, k0, i)

			# calc new transition layer
			transion_layer = self.get_transition_kernel(block_num)
			new_transion_layer = self.adjust_bottleneck_kernel(transion_layer, original_to_comprese, k0, len(bottleneck_kernels))


	def cluster(self):
		new_kernels = [[None] * self.densenet_model.layers_per_block] * self.densenet_model.total_blocks
		for block_num in range(self.densenet_model.total_blocks): 
			composite_kernels = self.gather_all_kernels_block('composite_function', block_num)
			for index, composite_kernel in enumerate(composite_kernels):
				cluster_centers, cluster_indices = self.cluster_composite_layer(composite_kernel)
				new_kernel_vector = np.array([cluster_centers[:,:,:,j] for j in cluster_indices])
				o, h, w, i = new_kernel_vector.shape 
				new_kernel_vector = np.reshape(new_kernel_vector, [h, w, i, o])
				new_kernels[block_num][index] = tf.convert_to_tensor(new_kernel_vector)
		return new_kernels
