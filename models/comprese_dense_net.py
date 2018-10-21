import os
import time
import sys
import shutil
from datetime import timedelta
from functools import reduce

import numpy as np
import tensorflow as tf

K_MEANS_ITERATION_NUM = 20
K0 = [24, 108, 150]

class CompreseDenseNet:
	def __init__(self, densenet_model, new_growth_rate):
		self.densenet_model = densenet_model
		self.new_growth_rate = new_growth_rate
		self.kmean =  tf.contrib.factorization.KMeansClustering(num_clusters=new_growth_rate, use_mini_batch=False)
		self.graph = tf.get_default_graph()
		self.bottleneck_output_size = self.growth_rate*4

	def gather_all_kernels_block(self, kernel_type, block_num):
		all_kernel = []
		for layer_num in range(self.densenet_model.layers_per_block):
			name = 'Block_{block_num}/layer_{layer_num}/{kernel_type}/kernel:0'.format(
					block_num=block_num, layer_num=layer_num, kernel_type=kernel_type)
			kernel_tensor = self.graph.get_tensor_by_name(name)
			all_kernel.append(kernel_tensor)
		return all_kernel		

	def get_transition_kernel(self, block_num):
		name = 'Transition_after_block_{block_num}/composite_function/kernel:0'
		kernel_tensor = self.graph.get_tensor_by_name(name)
		return kernel_tensor

	def cluster_composite_layer(self, composite_layer):
		h, w, i, o = composite_layer.shape
		composite_layer = tf.reshape(composite_layer, [h*w*i, o])
		cluster_centers, cluster_indices = self.k_means(composite_layer)
		cluster_centers = tf.reshape(cluster_centers [h, w, i, self.new_growth_rate])
		return cluster_centers, cluster_indices

	def k_means(self, vector):
		def input_fn():
			return tf.train.limit_epochs(vector, num_epochs=1)
		previous_centers = None
		for _ in range(K_MEANS_ITERATION_NUM):
			self.kmean.train(input_fn)
			cluster_centers = self.kmeans.cluster_centers()
			if previous_centers is not None:
				print('delta:{}'.format(sum(cluster_centers - previous_centers)))
			previous_centers = cluster_centers
		cluster_indices = list(self.kmeans.predict_cluster_index(input_fn))
		return cluster_centers, cluster_indices

	def create_orignatl_to_comprese_matrix(self, cluster_indices):
		assert len(cluster_indices) == self.growth_rate
		OtC = np.zeros((self.new_growth_rate, self.growth_rate, self.bottleneck_output_size))
		for i in cluster_indices:
			OtC[cluster_indices[i],[i],:] = 1
		OtC = tf.convert_to_tensor(OtC, dtype=tf.float32)
		return OtC

	def adjust_bottleneck_kernel(self, bottleneck_kernel, original_to_comprese, k0, layer_num):
		# don't touch k0 first params
		adjust_bottleneck = bottleneck_kernel[:][:][:k0][:]

		# for every input unite his wights by clusters
		for i in ragne(layer_num):
			for j in i:
				start_index = k0 + j*self.growth_rate
				adjust_bottleneck_slice = tf.matmul(original_to_comprese[j], bottleneck_kernel[:][:][start_index : start_index + self.growth_rate][:])
				adjust_bottleneck = tf.concat(axis=3, values=(adjust_bottleneck, adjust_bottleneck_slice))
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








