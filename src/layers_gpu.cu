#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include "gpu_matrix.h"
#include "cuda_utility.h"
#include "cuda_profile.h"
#include "utility.h"
#include "layers_gpu.h"
#include "kernels_gpu.h"

#define BLOCK_DIM_1D_X dim3(256, 1, 1)
#define BLOCK_DIM_1D_Y dim3(1, 256, 1)
#define BLOCK_DIM_2D dim3(KERNEL_2D_BLOCK_DIM, KERNEL_2D_BLOCK_DIM, 1)

static dim3 GetGridDim(GPUMatrix m, dim3 blockDim)
{
	dim3 gridDim = dim3((m.cols + blockDim.x - 1) / blockDim.x,
						(m.rows + blockDim.y - 1) / blockDim.y);
	return gridDim;
}

void OpGPU::MatFillScalar(GPUMatrix &m_a, float value, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_fill_scalar << < gridDim, blockDim, sharedMem, stream>> >(m_a, value);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatVecAdd(const GPUMatrix &m_a, const GPUMatrix &v_b, GPUMatrix &m_c, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_vec_add<< < gridDim, blockDim, sharedMem, stream>> >(m_a, v_b, m_c);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatMul(const GPUMatrix &m_a, const GPUMatrix &m_b, GPUMatrix &m_c, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_c, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_mul_shared << < gridDim, blockDim, sharedMem, stream>> > (m_a, m_b, m_c);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatTranspose(const GPUMatrix &m_a, GPUMatrix &m_a_t, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_a_t, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_transpose << < gridDim, blockDim, sharedMem, stream>> >(m_a, m_a_t);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatMatAdd(const GPUMatrix &m_a, const GPUMatrix &m_b, GPUMatrix &m_c, float beta, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_c, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_mat_add << < gridDim, blockDim, sharedMem, stream>> >(m_a, m_b, m_c, beta);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatColSum(const GPUMatrix &m_a, GPUMatrix &v_row_b, cudaStream_t stream)
{
	MatFillScalar(v_row_b, 0.0f, stream);

	dim3 blockDim = BLOCK_DIM_1D_X;
	dim3 gridDim = GetGridDim(v_row_b, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_col_sum << < gridDim, blockDim, sharedMem, stream>> >(m_a, v_row_b);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::VecRowSum(const GPUMatrix &v_col, float *d_sum, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_1D_Y;
	dim3 gridDim = GetGridDim(v_col, blockDim);
	size_t sharedMem = sizeof(float) * blockDim.y;
	Kernel::vec_row_sum << < gridDim, blockDim, sharedMem, stream>> >(v_col, d_sum);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatRowMax(const GPUMatrix &m_a, GPUMatrix &v_max, cudaStream_t stream)
{
	MatFillScalar(v_max, -CUDA_FLT_MAX, stream);

	dim3 blockDim = BLOCK_DIM_1D_X;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = sizeof(float) * blockDim.x;
	Kernel::mat_row_max <<< gridDim, blockDim, sharedMem, stream>> > (m_a, v_max);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatRowSum(const GPUMatrix &m_a, GPUMatrix &m_sum, cudaStream_t stream)
{
	MatFillScalar(m_sum, 0.0f, stream);

	dim3 blockDim = BLOCK_DIM_1D_X;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = sizeof(float) * blockDim.x;
	Kernel::mat_row_sum << < gridDim, blockDim, sharedMem, stream>> > (m_a, m_sum);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatSubMaxExp(const GPUMatrix &m_a, const GPUMatrix &m_max, GPUMatrix &m_exp, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_exp, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_exp<< < gridDim, blockDim, sharedMem, stream>> >(m_a, m_max, m_exp);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatSquare(const GPUMatrix &m_a, GPUMatrix &m_sqr, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_square << < gridDim, blockDim, sharedMem, stream>> >(m_a, m_sqr);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatScale(GPUMatrix m_a, float scale, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_a, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_scale << < gridDim, blockDim, sharedMem, stream>> >(m_a, scale);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatVecDivRowwise(const GPUMatrix &m_a, const GPUMatrix &v_col_b, GPUMatrix &m_c, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(m_c, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_vec_div_rowwise << < gridDim, blockDim, sharedMem, stream>> >(m_a, v_col_b, m_c);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::MatAddScalarOneHot(GPUMatrix &m_a, const GPUMatrix &v_col_one_hot, float scalar, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = GetGridDim(v_col_one_hot, blockDim);
	size_t sharedMem = 0;
	Kernel::mat_add_scalar_one_hot << < gridDim, blockDim, sharedMem, stream>> >(m_a, v_col_one_hot, scalar);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::CrossEntropyPerExampleLoss(const GPUMatrix &m_exp_scores, const GPUMatrix &m_exp_scores_sums, const GPUMatrix &v_col_y_correct,
										  GPUMatrix &v_col_example_losses, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_1D_Y;
	dim3 gridDim = GetGridDim(v_col_example_losses, blockDim);
	size_t sharedMem = 0;
	Kernel::cross_entropy_per_example_loss << < gridDim, blockDim, sharedMem, stream>> >(m_exp_scores, m_exp_scores_sums, v_col_y_correct, v_col_example_losses);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void OpGPU::ScalarScale(float *scalar, float scale, cudaStream_t stream)
{
	dim3 blockDim = 1;
	dim3 gridDim = 1;
	size_t sharedMem = 0;
	Kernel::scalar_scale << <gridDim, blockDim, sharedMem, stream>> >(scalar, scale);
	CUDAUtil::DEBUGSynchronizeDevice();
}

void LayerGPU::CrossEntropyLossForward(const GPUMatrix &m_scores, const GPUMatrix &v_col_y_correct, GPUMatrix &v_col_maxes, GPUMatrix &m_exp_scores,
									   GPUMatrix &v_col_exp_scores_sums, float *loss, cudaStream_t stream)
{
	OpGPU::MatRowMax(m_scores, v_col_maxes, stream);
	OpGPU::MatSubMaxExp(m_scores, v_col_maxes, m_exp_scores, stream);
	
	OpGPU::MatRowSum(m_exp_scores, v_col_exp_scores_sums, stream);
	GPUMatrix &v_col_example_losses = v_col_maxes;				// Reuse this matrix
	OpGPU::CrossEntropyPerExampleLoss(m_exp_scores, v_col_exp_scores_sums, v_col_y_correct, v_col_example_losses, stream);
	OpGPU::VecRowSum(v_col_example_losses, loss, stream);

	OpGPU::ScalarScale(loss, 1.0f / v_col_example_losses.rows, stream);
}

void LayerGPU::CrossEntropyLossBackward(const GPUMatrix &m_exp_scores, const GPUMatrix &v_col_exp_scores_sums, const GPUMatrix &v_col_y_correct,
											GPUMatrix &m_bottom_grad_scores, cudaStream_t stream)
{
	OpGPU::MatVecDivRowwise(m_exp_scores, v_col_exp_scores_sums, m_bottom_grad_scores, stream);
	OpGPU::MatAddScalarOneHot(m_bottom_grad_scores, v_col_y_correct, -1.0f, stream);
	OpGPU::MatScale(m_bottom_grad_scores, 1.0f / float(m_bottom_grad_scores.rows), stream);
}

// m_tmp must have enough memory to accomodate the largest of the reg_weights matrices
void LayerGPU::L2RegularizedLoss(std::vector<GPUMatrix> reg_weights, float *d_reg_loss, float reg_strength, 
								 GPUMatrix &m_tmp, GPUMatrix &v_tmp, cudaStream_t stream)
{
	// Sum the values of the squared weights
	for (size_t i = 0; i < reg_weights.size(); ++i)
	{
		m_tmp.rows = reg_weights[i].rows;
		m_tmp.cols = reg_weights[i].cols;
		v_tmp.rows = reg_weights[i].rows;
		v_tmp.cols = 1;
		
		OpGPU::MatSquare(reg_weights[i], m_tmp, stream);
		OpGPU::MatRowSum(m_tmp, v_tmp, stream);
		OpGPU::VecRowSum(v_tmp, d_reg_loss, stream);
	}

	OpGPU::ScalarScale(d_reg_loss, 0.5f * reg_strength, stream);
}

void LayerGPU::AffineLayerForward(GPUMatrix const &bottom_a, GPUMatrix const &bottom_weights, GPUMatrix const &bottom_bias,
	GPUMatrix &top_z, cudaStream_t stream)
{
	OpGPU::MatMul(bottom_a, bottom_weights, top_z, stream);
	OpGPU::MatVecAdd(top_z, bottom_bias, top_z, stream);
}

void LayerGPU::AffineLayerBackward(GPUMatrix &bottom_grad_a, GPUMatrix &bottom_grad_w, GPUMatrix &bottom_grad_bias,
	GPUMatrix const &bottom_a, GPUMatrix &bottom_a_trans, GPUMatrix const &bottom_w, GPUMatrix &bottom_w_trans,
	GPUMatrix const &top_grad, float regularizationStrength, cudaStream_t stream)
{
	// Grad w
	assert(bottom_a.realElementCount <= bottom_a_trans.realElementCount);
	bottom_a_trans.cols = bottom_a.rows;	// Reshape because bottom_a_trans is just helper memory. 
	bottom_a_trans.rows = bottom_a.cols;	// Swap rows and cols because of transpose.
	OpGPU::MatTranspose(bottom_a, bottom_a_trans, stream);

	OpGPU::MatMul(bottom_a_trans, top_grad, bottom_grad_w, stream);
	OpGPU::MatMatAdd(bottom_grad_w, bottom_w, bottom_grad_w, regularizationStrength, stream);

	// Grad a
	assert(bottom_w.realElementCount <= bottom_w_trans.realElementCount);
	bottom_w_trans.cols = bottom_w.rows;	// Reshape because bottom_a_trans is just helper memory
	bottom_w_trans.rows = bottom_w.cols;
	OpGPU::MatTranspose(bottom_w, bottom_w_trans, stream);

	OpGPU::MatMul(top_grad, bottom_w_trans, bottom_grad_a, stream);

	// Grad bias	
	OpGPU::MatColSum(top_grad, bottom_grad_bias, stream);
}

void LayerGPU::ActivationLayerForward(GPUMatrix const &bottom_z, GPUMatrix &top_a, ActivationFunction activation, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = dim3((top_a.cols + blockDim.x - 1) / blockDim.x,
						(top_a.rows + blockDim.y - 1) / blockDim.y);
	size_t sharedMem = 0;
	switch (activation)
	{
		case ReLU:
			Kernel::relu_layer_forward << < gridDim, blockDim, sharedMem, stream>> >(bottom_z, top_a);
			break;
		case Sigmoid:
			Kernel::sigmoid_layer_forward << < gridDim, blockDim, sharedMem, stream>> >(bottom_z, top_a);
			break;
		default:
			break;
	}
	CUDAUtil::DEBUGSynchronizeDevice();
}

void LayerGPU::ActivationLayerBackward(GPUMatrix &bottom_grad_z, GPUMatrix const &bottom_z, GPUMatrix const &top_grad_a, ActivationFunction activation, cudaStream_t stream)
{
	dim3 blockDim = BLOCK_DIM_2D;
	dim3 gridDim = dim3((bottom_grad_z.cols + blockDim.x - 1) / blockDim.x,
		(bottom_grad_z.rows + blockDim.y - 1) / blockDim.y);
	size_t sharedMem = 0;
	switch (activation)
	{
		case ReLU:
			Kernel::relu_layer_backward << < gridDim, blockDim, sharedMem, stream>> >(bottom_grad_z, bottom_z, top_grad_a);
			break;
		case Sigmoid:
			Kernel::sigmoid_layer_backward<< < gridDim, blockDim, sharedMem, stream>> >(bottom_grad_z, bottom_z, top_grad_a);
			break;
		default:
			break;
	}
	CUDAUtil::DEBUGSynchronizeDevice();
}

void ProfileLayerGPU::MatMul(const GPUMatrix &a, const GPUMatrix &b, GPUMatrix &c)
{
	{
		dim3 blockDim = BLOCK_DIM_2D;
		dim3 gridDim = dim3((c.cols + blockDim.x - 1) / blockDim.x,
					   (c.rows + blockDim.y - 1) / blockDim.y);
		size_t sharedMemBytes = 0;
		CUDAUtil::ProfileFunction(5, Kernel::mat_mul_global, "mat_mul_global", gridDim, blockDim, sharedMemBytes, a, b, c);
	}

	{
		dim3 blockDim = BLOCK_DIM_2D;
		dim3 gridDim = GetGridDim(a, blockDim);
		size_t sharedMemBytes = 0;
		CUDAUtil::ProfileFunction(5, Kernel::mat_mul_shared, "mat_mul_shared", gridDim, blockDim, sharedMemBytes, a, b, c);
	}
}

void ProfileLayerGPU::MatExp(const GPUMatrix &m_a, const GPUMatrix &m_max, GPUMatrix &m_exp)
{
	int numRuns = 25;
	dim3 blockDim = BLOCK_DIM_1D_X;
	dim3 gridDim = dim3((m_exp.cols + blockDim.x - 1) / blockDim.x,
		(m_exp.rows + blockDim.y - 1) / blockDim.y);
	size_t sharedMemBytes = 0;
	CUDAUtil::ProfileFunction(numRuns, Kernel::mat_exp, "mat_exp", gridDim, blockDim, sharedMemBytes, m_a, m_max, m_exp);
}