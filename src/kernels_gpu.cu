#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

#include "utility.h"
#include "kernels_gpu.h"
#include "gpu_matrix.h"

__device__
int2 GetGlobalIndex()
{
	return make_int2(blockIdx.x * blockDim.x + threadIdx.x,
					 blockIdx.y * blockDim.y + threadIdx.y);
}

__global__
void Kernel::mat_fill_scalar(GPUMatrix m_a, float fill_value)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols && gIdx.y < m_a.rows)
	{
		m_a.data[gIdx.y * m_a.cols + gIdx.x] = fill_value;
	}
}

__global__
void Kernel::scalar_scale(float *scalar, float scale)
{
	*scalar *= scale;
}

__global__
void Kernel::mat_mul_global(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_c.cols && gIdx.y < m_c.rows)
	{
		float dot = 0.0f;
		for (int i = 0; i < m_a.cols; ++i)
		{
			float aValue = m_a.data[m_a.cols * gIdx.y + i];
			float bValue = m_b.data[m_b.cols * i + gIdx.x];
			dot += aValue * bValue;
		}
		m_c.data[gIdx.y * m_c.cols + gIdx.x] = dot;
	}
}

/*
	Tiled matrix multiplication. Each block of threads in m_c calculates the complete dot product
	by indexing a sliding window in a (moving horizontally to the right) and in b (moving vertically
	down). The values in both windows are stored in shared memory and then dot producted together.
	This partial result is added to an accumulator and the windows shift again.
*/
__global__
void Kernel::mat_mul_shared(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c)
{
	const int BLOCK_SIZE = KERNEL_2D_BLOCK_DIM;
	__shared__ float Ashared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bshared[BLOCK_SIZE][BLOCK_SIZE];

	int2 gIdx = GetGlobalIndex();

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aWidth = m_a.cols;
	int aStart = blockIdx.y * BLOCK_SIZE * aWidth;	// Move horizontally to the right in a
	int aEnd = aStart + aWidth - 1;
	int aStep = BLOCK_SIZE;

	int bWidth = m_b.cols;							// Move vertically down in b
	int bStart = blockIdx.x * BLOCK_SIZE;
	int bEnd = bStart + bWidth * (m_b.rows - 1);
	int bStep = BLOCK_SIZE * bWidth;

	float cValue = 0.0f;
	for (int a = aStart, b = bStart; a <= aEnd; a += aStep, b += bStep)
	{
		// Initialize shared memory
		if (a + tx <= aEnd && gIdx.y < m_c.rows)
			Ashared[ty][tx] = m_a.data[a + ty * aWidth + tx];
		else
			Ashared[ty][tx] = 0.0f;

		if (b + ty <= bEnd && gIdx.x < m_c.cols)
			Bshared[ty][tx] = m_b.data[b + ty * bWidth + tx];
		else
			Bshared[ty][tx] = 0.0f;
		__syncthreads();

		// Add partial dot product
		for (int i = 0; i < BLOCK_SIZE; ++i)
			cValue += Ashared[ty][i] * Bshared[i][tx];
		__syncthreads();
	}

	if (gIdx.x < m_c.cols && gIdx.y < m_c.rows)
		m_c.data[gIdx.y * m_c.cols + gIdx.x] = cValue;
}

__global__
void Kernel::mat_vec_add(const GPUMatrix m_a, const GPUMatrix v_b, GPUMatrix m_c)
{
	assert(m_a.cols == m_c.cols);
	assert(m_a.rows == m_c.rows);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_c.cols && gIdx.y < m_c.rows)
	{
		int idx = gIdx.y * m_c.cols + gIdx.x;
		m_c.data[idx] = m_a.data[idx] + v_b.data[gIdx.x];
	}
}

// TODO: Faster transpose
__global__
void Kernel::mat_transpose(const GPUMatrix m_a, GPUMatrix m_t)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_t.cols && gIdx.y < m_t.rows)
	{
		m_t.data[gIdx.y * m_t.cols + gIdx.x] = m_a.data[gIdx.x * m_a.cols + gIdx.y];
	}
}

__global__
void Kernel::mat_mat_add(const GPUMatrix m_a, const GPUMatrix m_b, GPUMatrix m_c, float beta)
{
	assert(m_a.rows == m_b.rows);
	assert(m_a.cols == m_b.cols);
	assert(m_a.cols == m_c.cols);
	assert(m_a.rows == m_c.rows);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_c.cols && gIdx.y < m_c.rows)
	{
		int idx = gIdx.y * m_c.cols + gIdx.x;
		m_c.data[idx] = m_a.data[idx] + beta * m_b.data[idx];
	}
}

// TODO: speed up
__global__
void Kernel::mat_col_sum(const GPUMatrix m_a, GPUMatrix v_row_b)
{
	assert(m_a.cols == v_row_b.cols);
	assert(v_row_b.rows == 1);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < v_row_b.cols)
	{
		for (int i = 0; i < m_a.rows; ++i)
		{
			//v_row_b.data[gIdx.x] += m_a.data[i * m_a.cols + gIdx.x];
			atomicAdd(&v_row_b.data[gIdx.x], m_a.data[i * m_a.cols + gIdx.x]);
		}
	}
}

__global__
void Kernel::relu_layer_forward(const GPUMatrix bottom_z, GPUMatrix top_a)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < top_a.cols && gIdx.y < top_a.rows)
	{
		int idx = gIdx.y * top_a.cols + gIdx.x;
		top_a.data[idx] = bottom_z.data[idx] > 0 ? bottom_z.data[idx] : 0;
	}
}

__global__
void Kernel::relu_layer_backward(GPUMatrix bottom_grad_z, const GPUMatrix bottom_z, const GPUMatrix top_grad)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < bottom_grad_z.cols && gIdx.y < bottom_grad_z.rows)
	{
		int idx = gIdx.y * bottom_grad_z.cols + gIdx.x;
		bottom_grad_z.data[idx] = bottom_z.data[idx] > 0.0f ? 1.0f * top_grad.data[idx] : 0.0f;
	}
}

__global__
void Kernel::sigmoid_layer_forward(const GPUMatrix bottom_z, GPUMatrix top_a)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < top_a.cols && gIdx.y < top_a.rows)
	{
		int idx = gIdx.y * top_a.cols + gIdx.x;
		top_a.data[idx] = 1.0f / (1.0f + __expf(-bottom_z.data[idx]));
	}
}

__global__
void Kernel::sigmoid_layer_backward(GPUMatrix bottom_grad_z, const GPUMatrix bottom_z, const GPUMatrix top_grad)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < bottom_grad_z.cols && gIdx.y < bottom_grad_z.rows)
	{
		int idx = gIdx.y * bottom_grad_z.cols + gIdx.x;
		float sigmoid = 1.0f / (1.0f + __expf(-bottom_z.data[idx]));
		bottom_grad_z.data[idx] = sigmoid * (1.0f - sigmoid) * top_grad.data[idx];
	}
}

// see https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ 
static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__global__
void Kernel::mat_row_max(const GPUMatrix m_a, GPUMatrix v_col_b)
{
	extern __shared__ float shared[];

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols)
		shared[threadIdx.x] = m_a.data[gIdx.y * m_a.cols + gIdx.x];
	else
		shared[threadIdx.x] = -CUDA_FLT_MAX;
	__syncthreads();

	for (int size = blockDim.x >> 1; size >= 1; size >>= 1)
	{
		if (threadIdx.x < size)
		{
			shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + size]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		atomicMax(&v_col_b.data[gIdx.y], shared[0]);
}

__global__
void Kernel::mat_row_sum(const GPUMatrix m_a, GPUMatrix v_col_b)
{
	extern __shared__ float shared[];

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols)
		shared[threadIdx.x] = m_a.data[gIdx.y * m_a.cols + gIdx.x];
	else
		shared[threadIdx.x] = 0.0f;
	__syncthreads();

	for (int size = blockDim.x >> 1; size >= 1; size >>= 1)
	{
		if (threadIdx.x < size)
			shared[threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x + size];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		atomicAdd(&v_col_b.data[gIdx.y], shared[0]);
}

__global__
void Kernel::vec_row_sum(const GPUMatrix v, float *sum)
{
	extern __shared__ float shared[];

	int2 gIdx = GetGlobalIndex();
	if (gIdx.y < v.rows)
		shared[threadIdx.y] = v.data[gIdx.y];
	else
		shared[threadIdx.y] = 0.0f;
	__syncthreads();

	for (int size = blockDim.y >> 1; size >= 1; size >>= 1)
	{
		if (threadIdx.y < size)
		{
			shared[threadIdx.y] = shared[threadIdx.y] + shared[threadIdx.y + size];
		}
		__syncthreads();
	}

	if (threadIdx.y == 0)
		atomicAdd(sum, shared[0]);
}

__global__
void Kernel::mat_exp(const GPUMatrix m_a, const GPUMatrix v_col_max, GPUMatrix m_exp)
{
	assert(m_a.cols == m_exp.cols);
	assert(m_a.rows == m_exp.rows);
	assert(m_a.rows == v_col_max.rows);
	assert(v_col_max.cols == 1);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols && gIdx.y < m_a.rows)
	{
		// Subtract row maximums and exponentiate
		int idx = gIdx.y * m_a.cols + gIdx.x;
		m_exp.data[idx] = __expf(m_a.data[idx] - v_col_max.data[gIdx.y]);
	}
}

__global__
void Kernel::cross_entropy_per_example_loss(const GPUMatrix m_exp_scores, const GPUMatrix m_exp_scores_sums, const GPUMatrix m_y_correct,
											GPUMatrix v_col_example_losses)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.y < m_exp_scores.rows)
	{
		int correctYColumn = int(m_y_correct.data[gIdx.y]);
		float correctClassScore = m_exp_scores.data[gIdx.y * m_exp_scores.cols + correctYColumn];
		float rowExpSum = m_exp_scores_sums.data[gIdx.y];
		float exampleLoss = -__logf(correctClassScore / rowExpSum);	
		v_col_example_losses.data[gIdx.y] = exampleLoss;
	}
}

__global__
void Kernel::cross_entropy_total_loss(const GPUMatrix m_exp_scores, const GPUMatrix m_exp_scores_sums, const GPUMatrix m_y_correct,
									  float *total_loss)
{
	extern __shared__ float shared[];
	
	int2 gIdx = GetGlobalIndex();	
	if (gIdx.y < m_exp_scores.rows)
	{
		int correctYColumn = int(m_y_correct.data[gIdx.y]);
		float correctClassScore = m_exp_scores.data[gIdx.y * m_exp_scores.cols + correctYColumn];
		float rowExpSum = m_exp_scores_sums.data[gIdx.y];
		float exampleLoss = -__logf(correctClassScore / rowExpSum);
		shared[threadIdx.y] = exampleLoss;
	}
	else
	{
		shared[threadIdx.y] = 0.0f;
	}
	__syncthreads();

	for (int size = blockDim.y >> 1; size >= 1; size >>= 1)
	{
		if (threadIdx.y < size)
		{
			shared[threadIdx.y] = shared[threadIdx.y] + shared[threadIdx.y + size];
		}
		__syncthreads();
	}

	if (threadIdx.y == 0)
		atomicAdd(total_loss, shared[0]);
}

__global__
void Kernel::mat_square(const GPUMatrix m_a, GPUMatrix m_sqr)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols && gIdx.y < m_a.rows)
	{
		int idx = gIdx.y * m_a.cols + gIdx.x;
		m_sqr.data[idx] = m_a.data[idx] * m_a.data[idx];
	}
}

__global__
void Kernel::mat_vec_div_rowwise(const GPUMatrix m_a, const GPUMatrix v_col_b, GPUMatrix m_c)
{
	assert(m_a.cols == m_c.cols && m_a.rows == m_c.rows);
	assert(v_col_b.cols == 1 && v_col_b.rows == m_a.rows);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols && gIdx.y < m_a.rows)
	{
		int idx = gIdx.y * m_a.cols + gIdx.x;
		m_c.data[idx] = m_a.data[idx] / v_col_b.data[gIdx.y];
	}
}

__global__
void Kernel::mat_add_scalar_one_hot(GPUMatrix m_a, const GPUMatrix v_col_one_hot, float scalar)
{
	assert(v_col_one_hot.cols == 1 && v_col_one_hot.rows == m_a.rows);

	int2 gIdx = GetGlobalIndex();
	if (gIdx.y < m_a.rows)
	{
		int hotColumn = int(v_col_one_hot.data[gIdx.y]);
		assert(hotColumn < m_a.cols);
		int idx = gIdx.y * m_a.cols + hotColumn;
		m_a.data[idx] += scalar;
	}
}

__global__
void Kernel::mat_scale(GPUMatrix m_a, float alpha)
{
	int2 gIdx = GetGlobalIndex();
	if (gIdx.x < m_a.cols && gIdx.y < m_a.rows)
	{
		int idx = gIdx.y * m_a.cols + gIdx.x;
		m_a.data[idx] = alpha * m_a.data[idx];
	}
}