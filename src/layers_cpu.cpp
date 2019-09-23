#include <vector>
#include <iostream>

#include "layers_cpu.h"

#define ASSERT_DIMS_EQ(a, b) \
	assert((a).cols() == (b).cols()); \
	assert((a).rows() == (b).rows()); 

float ReLUForward(float x)
{
	return x > 0 ? x : 0;
}

float ReLUBackward(float bottom, float top_grad)
{
	return bottom > 0 ? 1.0f * top_grad : 0.0f;
}

float SigmoidForward(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float SigmoidBackward(float bottom, float top_grad)
{
	float sigmoid = SigmoidForward(bottom);
	return sigmoid * (1.0f - sigmoid) * top_grad;
}

void LayerCPU::CrossEntropyLossForward(const EMatrix &bottom_scores, const EMatrix &bottom_ycorrect, EMatrix &exp_scores, 
										EMatrix &exp_scores_sums, float &top_loss)
{
	assert(bottom_scores.rows() == bottom_ycorrect.rows());
	assert(bottom_ycorrect.cols() == 1);

	for (int r = 0; r < bottom_scores.rows(); ++r)
	{
		EMatrix scoresRow = bottom_scores.block(r, 0, 1, bottom_scores.cols());
		float maxValue = scoresRow.maxCoeff();
		for (int c = 0; c < bottom_scores.cols(); ++c)
		{
			exp_scores(r, c) = expf(bottom_scores(r,c) - maxValue);														// Subtract max value for numerical stability
		}
	}

	for (int r = 0; r < bottom_scores.rows(); ++r)
	{
		EMatrix expScoresRow = exp_scores.block(r, 0, 1, exp_scores.cols());
		exp_scores_sums(r, 0) = expScoresRow.sum();
	}

	EMatrix perExampleLosses(bottom_scores.rows(), 1);
	for (int r = 0; r < bottom_scores.rows(); ++r)
	{
		perExampleLosses(r, 0) = exp_scores(r, int(bottom_ycorrect(r, 0))) / exp_scores_sums(r, 0);	// Correct class probabilities
		perExampleLosses(r, 0) = -logf(perExampleLosses(r, 0));
	}

	top_loss = perExampleLosses.mean();
}

void LayerCPU::L2RegularizedLoss(float &l2_loss, float reg_strength, std::vector<EMatrix> const &regularization_weights)
{
	l2_loss = 0.0f;
	for (unsigned i = 0; i < regularization_weights.size(); ++i)
	{
		EMatrix weightSquared = regularization_weights[i].array().pow(2.0f);
		l2_loss += weightSquared.sum();
	}
	l2_loss *= 0.5f * reg_strength;
}

void LayerCPU::CrossEntropyLossBackward(EMatrix &bottom_grad_scores, const EMatrix &exp_scores, const EMatrix &exp_scores_sums, const EMatrix &bottom_ycorrect)
{
	ASSERT_DIMS_EQ(exp_scores, bottom_grad_scores);
	assert(exp_scores.rows() == bottom_ycorrect.rows());
	assert(bottom_ycorrect.cols() == 1);

	for (int r = 0; r < bottom_grad_scores.rows(); ++r)
	{
		for (int c = 0; c < bottom_grad_scores.cols(); ++c)
		{
			bottom_grad_scores(r, c) = exp_scores(r, c) / exp_scores_sums(r, 0);
		}
		bottom_grad_scores(r, int(bottom_ycorrect(r, 0))) += -1.0;
	}

	bottom_grad_scores = bottom_grad_scores.array() / bottom_grad_scores.rows();
}

void LayerCPU::AffineLayerForward(const EMatrix &bottom_a, const EMatrix &bottom_weights, const EMatrix &bottom_bias, EMatrix &top_z)
{
	assert(bottom_a.cols() == bottom_weights.rows());
	assert(top_z.rows() == bottom_a.rows());
	assert(top_z.cols() == bottom_weights.cols());
	assert(bottom_bias.rows() == 1);
	assert(bottom_bias.cols() == bottom_weights.cols());

	top_z = bottom_a * bottom_weights;
	top_z = top_z.rowwise() + bottom_bias.row(0);
}

void LayerCPU::AffineLayerBackward(EMatrix &bottom_grad_a, EMatrix &bottom_grad_w, EMatrix &bottom_grad_bias, 
								const EMatrix &bottom_a, const EMatrix &bottom_w, const EMatrix &top_grad, float regularization)
{
	assert(bottom_grad_w.rows() == bottom_a.cols());
	assert(bottom_grad_bias.rows() == 1 && bottom_grad_bias.cols() == top_grad.cols());

	bottom_grad_w = bottom_a.transpose() * top_grad;
	bottom_grad_w = bottom_grad_w + regularization * bottom_w;

	bottom_grad_a = top_grad * bottom_w.transpose();
	
	bottom_grad_bias = bottom_grad_bias.setZero();
	for (int r = 0; r < top_grad.rows(); ++r)
	{
		for (int c = 0; c < top_grad.cols(); ++c)
		{
			bottom_grad_bias(0, c) += top_grad(r, c);
		}
	}
}

void LayerCPU::ActivationLayerForward(const EMatrix &bottom_z, EMatrix &top_a, ActivationFunction activation)
{
	ASSERT_DIMS_EQ(bottom_z, top_a);

	float (*forwardFunction)(float);
	switch (activation)
	{
		case ReLU:
			forwardFunction = ReLUForward;
			break;
		case Sigmoid:
			forwardFunction = SigmoidForward;
			break;
	}

	for (int r = 0; r < bottom_z.rows(); ++r)
	{
		for (int c = 0; c < bottom_z.cols(); ++c)
		{
			top_a(r, c) = forwardFunction(bottom_z(r, c));
		}
	}
}

void LayerCPU::ActivationLayerBackward(EMatrix &bottom_grad_z, const EMatrix &bottom_z, const EMatrix &top_grad, ActivationFunction activation)
{
	ASSERT_DIMS_EQ(bottom_z, bottom_grad_z);
	ASSERT_DIMS_EQ(bottom_z, top_grad);

	float (*backwardFunction)(float, float);
	switch (activation)
	{
		case ReLU:
			backwardFunction = ReLUBackward;
			break;
		case Sigmoid:
			backwardFunction = SigmoidBackward;
			break;
	}

	for (int r = 0; r < bottom_z.rows(); ++r)
	{
		for (int c = 0; c < bottom_z.cols(); ++c)
		{
			bottom_grad_z(r, c) = bottom_z(r,c) > 0.0f ? 1.0f * top_grad(r,c) : 0.0f;
		}
	}
}