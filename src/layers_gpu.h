#pragma once

#include <vector>

#include "activation_functions.h"
#include "cuda_utility.h"

class GPUMatrix;

namespace OpGPU
{
	void MatFillScalar(GPUMatrix &m_a, float value, cudaStream_t stream);
	void MatMul(const GPUMatrix &a, const GPUMatrix &b, GPUMatrix &c, cudaStream_t stream);
	void MatTranspose(const GPUMatrix &m_a, GPUMatrix &m_a_t, cudaStream_t stream);
	void MatMatAdd(const GPUMatrix &m_a, const GPUMatrix &m_b, GPUMatrix &m_c, float beta, cudaStream_t stream);
	void MatColSum(const GPUMatrix &m_a, GPUMatrix &v_b, cudaStream_t stream);
	void VecRowSum(const GPUMatrix &v_col, float *sum, cudaStream_t stream);
	void MatVecAdd(const GPUMatrix &m_a, const GPUMatrix &v_b, GPUMatrix &m_c, cudaStream_t stream);
	void MatVecDivRowwise(const GPUMatrix &m_a, const GPUMatrix &v_col_b, GPUMatrix &m_c, cudaStream_t stream);
	void MatAddScalarOneHot(GPUMatrix &m_a, const GPUMatrix &v_col_one_hot, float scalar, cudaStream_t stream);
	void MatRowMax(const GPUMatrix &m_a, GPUMatrix &m_max, cudaStream_t stream);
	void MatRowSum(const GPUMatrix &m_a, GPUMatrix &m_sum, cudaStream_t stream);
	void MatSubMaxExp(const GPUMatrix &m_a, const GPUMatrix &m_max, GPUMatrix &m_exp, cudaStream_t stream);
	void MatSquare(const GPUMatrix &m_a, GPUMatrix &m_sqr, cudaStream_t stream);
	void MatScale(GPUMatrix m_a, float scale, cudaStream_t stream);
	void ScalarScale(float *scalar, float scale, cudaStream_t stream);
	void CrossEntropyPerExampleLoss(const GPUMatrix &m_exp_scores, const GPUMatrix &m_exp_scores_sums, const GPUMatrix &v_col_y_correct,
		GPUMatrix &v_col_example_losses, cudaStream_t stream);
}

namespace LayerGPU
{
	void ActivationLayerForward(GPUMatrix const &bottom_z, GPUMatrix &top_a, ActivationFunction activation, cudaStream_t stream);
	void ActivationLayerBackward(GPUMatrix &bottom_grad_z, GPUMatrix const &bottom_z, GPUMatrix const &top_grad_a, ActivationFunction activation, cudaStream_t stream);
	void AffineLayerForward(GPUMatrix const &bottom_a, GPUMatrix const &bottom_weights, GPUMatrix const &bottom_bias, 
							GPUMatrix &top_z, cudaStream_t stream);
	void AffineLayerBackward(GPUMatrix &bottom_grad_a, GPUMatrix &bottom_grad_w, GPUMatrix &bottom_grad_bias,
							 GPUMatrix const &bottom_a, GPUMatrix &bottom_a_trans, GPUMatrix const &bottom_w, GPUMatrix &bottom_w_trans, 
							 GPUMatrix const &top_grad, float regularizationStrength, cudaStream_t stream);
	void CrossEntropyLossForward(const GPUMatrix &m_scores, const GPUMatrix &v_col_y_correct, GPUMatrix &v_col_maxes, GPUMatrix &m_exp_scores,
								 GPUMatrix &v_col_exp_scores_sums, float *loss, cudaStream_t stream);
	void CrossEntropyLossBackward(const GPUMatrix &m_exp_scores, const GPUMatrix &m_exp_scores_sums, const GPUMatrix &m_y_correct,
								  GPUMatrix &m_bottom_grad_scores, cudaStream_t stream);
	void L2RegularizedLoss(std::vector<GPUMatrix> reg_weights, float *l2_loss, float reg_strength, GPUMatrix &m_tmp, GPUMatrix &v_tmp, cudaStream_t stream);
}

namespace ProfileLayerGPU
{
	void MatMul(const GPUMatrix &a, const GPUMatrix &b, GPUMatrix &c);
	void MatExp(const GPUMatrix &m_a, const GPUMatrix &m_max, GPUMatrix &m_exp);
}