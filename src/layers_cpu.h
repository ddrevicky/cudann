#pragma once

#include <vector>

#include "activation_functions.h"
#include "e_matrix.h"

namespace LayerCPU
{
	void AffineLayerForward(const EMatrix &bottom_a, const EMatrix &bottom_weights, const EMatrix &bottom_bias, EMatrix &top_z);
	void AffineLayerBackward(EMatrix &bottom_grad_a, EMatrix &bottom_grad_w, EMatrix &bottom_grad_bias,
							 const EMatrix &bottom_a, const EMatrix &bottom_weights, const EMatrix &top_grad, float reg_strength);
	void ActivationLayerForward(const EMatrix &bottom_z, EMatrix &top_a, ActivationFunction activation);
	void ActivationLayerBackward(EMatrix &bottom_grad_z, const EMatrix &bottom_z, const EMatrix &top_grad_a, ActivationFunction activation);
	void CrossEntropyLossForward(const EMatrix &bottom_scores, const EMatrix &bottom_ycorrect, EMatrix &exp_scores,
										 EMatrix &exp_scores_sums, float &top_loss);
	void CrossEntropyLossBackward(EMatrix &bottom_grad_scores, const EMatrix &exp_scores, const EMatrix &exp_scores_sums, const EMatrix &bottom_ycorrect);
	void L2RegularizedLoss(float &l2_loss, float reg_strength, std::vector<EMatrix> const &regularization_weights);

}