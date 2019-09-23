#pragma once

#include <cuda_runtime.h>

#define CUDA_FLT_MAX          3.402823466e+38F

#define KERNEL_2D_BLOCK_DIM 16

class GPUMatrix;

namespace Kernel
{
	__global__
	void mat_fill_scalar(GPUMatrix m_a, float fill_value);
	__global__
	void scalar_scale(float *scalar, float scale);
	__global__
	void mat_mul_shared(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c);
	__global__
	void mat_mul_global(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c);
	__global__
	void mat_scale(GPUMatrix m_a, float alpha);
	__global__
	void mat_vec_add(const GPUMatrix m_a, const GPUMatrix v_b, GPUMatrix m_c);
	__global__
	void mat_transpose(const GPUMatrix m_a, GPUMatrix m_t);
	__global__
	void mat_mat_add(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c, float beta);
	__global__
	void relu_layer_forward(const GPUMatrix bottom_z, GPUMatrix top_a);
	__global__
	void relu_layer_backward(GPUMatrix bottom_grad_z, const GPUMatrix bottom_z, const GPUMatrix top_grad);
	__global__
	void sigmoid_layer_forward(const GPUMatrix bottom_z, GPUMatrix top_a);
	__global__
	void sigmoid_layer_backward(GPUMatrix bottom_grad_z, const GPUMatrix bottom_z, const GPUMatrix top_grad);
	__global__
	void mat_row_max_naive(const GPUMatrix m_a, GPUMatrix v_max);
	__global__
	void mat_row_max(const GPUMatrix m_a, GPUMatrix v_max);
	__global__
	void mat_exp(const GPUMatrix m_a, const GPUMatrix m_max, GPUMatrix m_exp);
	__global__
	void mat_square(const GPUMatrix m_a, GPUMatrix m_sqr);
	__global__
	void mat_row_sum(const GPUMatrix m_a, GPUMatrix v_col_b);
	__global__
	void mat_col_sum(const GPUMatrix m_a, GPUMatrix v_row_b);
	__global__
	void vec_row_sum(const GPUMatrix v, float *sum);
	__global__
	void cross_entropy_per_example_loss(const GPUMatrix m_exp_scores, const GPUMatrix m_exp_scores_sums, const GPUMatrix m_y_correct,
										GPUMatrix v_col_example_losses);
	__global__
	void cross_entropy_total_loss(const GPUMatrix m_exp_scores, const GPUMatrix m_exp_scores_sums, const GPUMatrix m_y_correct,
						   		  float *total_loss);
	__global__
	void mat_add_scalar_one_hot(GPUMatrix m_a, const GPUMatrix v_col_one_hot, float scalar);
	__global__
	void mat_vec_div_rowwise(const GPUMatrix m_a, const GPUMatrix v_col_b, GPUMatrix m_c);
}