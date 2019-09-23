#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "neural_net_cpu.h"
#include "layers_cpu.h"
#include "e_matrix.h"

using std::to_string;

NeuralNetCPU::NeuralNetCPU()
{
}

NeuralNetCPU::NeuralNetCPU(std::vector<unsigned> sizes, ActivationFunction activation, float regStrength, float weightScale, int randSeed, int batchSize)
{
	UAssert(sizes.size() > 1, "Number of layers must be greater than 1.");
	UAssert(regStrength >= 0, "Regularization strength must be non-negative.");
	this->sizes = sizes;
	this->regularizationStrength = regStrength;
	this->batchSize = batchSize;
	this->activation = activation;

	std::srand(randSeed);
	std::default_random_engine generator(randSeed);
	std::normal_distribution<float> standardNormal(0.0f, 1.0f);

	// Init weights, biases and preallocate matrices for gradients
	const size_t numLayers = sizes.size();
	for (size_t layer = 0; layer < numLayers - 1; ++layer)
	{
		EMatrix weight(sizes[layer], sizes[layer + 1]);
		for (int r = 0; r < weight.rows(); ++r)
		{
			for (int c = 0; c < weight.cols(); ++c)
			{
				weight(r, c) = weightScale * standardNormal(generator) / sqrtf(sizes[layer]);
			}
		}
		weights.push_back(weight);
		gradWeights.push_back(EMatrix(sizes[layer], sizes[layer + 1]).setZero());
		biases.push_back(EMatrix(1, sizes[layer + 1]).setZero());
		gradBiases.push_back(EMatrix(1, sizes[layer + 1]).setZero());
	}
}

void NeuralNetCPU::ReportAccuracy(EMatrix &x, EMatrix &y)
{
	unsigned int flags = MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES;
	float learnRate = 0.0f;
	Prediction prediction = Loss(x, y, flags, learnRate);

	int correctPredictions = 0;
	for (int r = 0; r < y.rows(); ++r)
	{
		EMatrix singleExampleScores = prediction.scores.row(r);
		int maxRow, maxCol;
		singleExampleScores.maxCoeff(&maxRow, &maxCol);
		int predictedClass = maxCol;
		int correctClass = int(y(r, 0));

		if (predictedClass == correctClass)
			++correctPredictions;
	}

	float accuracy = float(correctPredictions) / float(y.rows());
	std::cout << "Loss " << prediction.loss << ", Accuracy " << 100 * accuracy << "%\n";
}

double NeuralNetCPU::SGD(EMatrix &XTrain, EMatrix &yTrain, int epochs, float learnRate, int decreaseLearnRateAfterEpochs, float decreaseFactor, bool verbose)
{
	double totalEpochTime = 0.0;
	int batchesProcessed = 0;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		std::cout << "Epoch " << epoch << " ";

		if (epoch % decreaseLearnRateAfterEpochs == 0 && epoch > 0)
		{
			learnRate *= decreaseFactor;
			std::cout << "Decreasing learning rate by a factor of " << decreaseFactor << ", new value " << learnRate << "\n";
		}
		// Shuffle training examples
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation(XTrain.rows());
		permutation.setIdentity();
		std::random_shuffle(permutation.indices().data(), permutation.indices().data() + permutation.indices().size());
		EMatrix xShuffled = permutation * XTrain;
		EMatrix yShuffled = permutation * yTrain;

		// Iterate over Minibatches 
		CPUTimer epochTimer;
		epochTimer.Start();
		int numBatches = (int(xShuffled.rows()) + batchSize - 1) / batchSize;
		for (int batchStart = 0; batchStart < XTrain.rows(); batchStart += batchSize)
		{
			int batchEnd = std::min(batchStart + batchSize, int(xShuffled.rows()));
			int currentBatchSize = batchEnd - batchStart;

			EMatrix xBatch = xShuffled.block(batchStart, 0, currentBatchSize, xShuffled.cols());
			EMatrix yBatch = yShuffled.block(batchStart, 0, currentBatchSize, yShuffled.cols());

			unsigned int modelFlags = MODEL_GRAD_UPDATE;
			++batchesProcessed;
			if (verbose && (batchesProcessed % numBatches == 0))
				modelFlags |= MODEL_PREDICT_LOSS;

			Prediction prediction = Loss(xBatch, yBatch, modelFlags, learnRate);

			if (verbose && (batchesProcessed % numBatches == 0))
				std::cout << "Loss " << prediction.loss;
		}
		std::cout << std::endl;
		epochTimer.Stop();
		totalEpochTime += epochTimer.GetElapsedTime();
	}

	return totalEpochTime / double(epochs);
}

Prediction NeuralNetCPU::Loss(EMatrix &X, EMatrix &y, unsigned int flags, float learnRate)
{
	/* Notation:
	   Z = X * W + b
	   A = activation(Z)
	*/
	size_t numComputingLayers = sizes.size() - 1;
	int batchSize = int(X.rows());
	
	if (ACache.size() == 0 || ACache[0].rows() != batchSize)
	{
		if (ACache.size() > 0 && ACache[0].rows() != batchSize)
		{
			ACache.clear();
			dACache.clear();
			ZCache.clear();
			dZCache.clear();
		}
		for (size_t layer = 0; layer < numComputingLayers; ++layer)
		{
			ACache.push_back(EMatrix(batchSize, sizes[layer]).setZero());
			dACache.push_back(EMatrix(batchSize, sizes[layer]).setZero());
			ZCache.push_back(EMatrix(batchSize, sizes[layer + 1]).setZero());
			dZCache.push_back(EMatrix(batchSize, sizes[layer + 1]).setZero());
		}
		expScoresCache = EMatrix(batchSize, sizes[numComputingLayers]).setZero();
		expScoresSumsCache = EMatrix(batchSize, 1).setZero();
	}

	// Forward
	ACache[0] = X;
	EMatrix *prevLayerA = &ACache[0];
	float crossEntropyLoss = 0.0f;
	float l2Loss = 0.0f;

	for (size_t layer = 0; layer < numComputingLayers; ++layer)
	{
		EMatrix &W = weights[layer];
		EMatrix &b = biases[layer];
		EMatrix &Z = ZCache[layer];

		LayerCPU::AffineLayerForward(*prevLayerA, W, b, Z);

		if (layer < numComputingLayers - 1)
		{
			LayerCPU::ActivationLayerForward(Z, ACache[layer + 1], this->activation);
			prevLayerA = &ACache[layer + 1];
		}
		else 
		{
			LayerCPU::CrossEntropyLossForward(Z, y, expScoresCache, expScoresSumsCache, crossEntropyLoss);
			LayerCPU::L2RegularizedLoss(l2Loss, regularizationStrength, weights);
		}
	}

	Prediction prediction;
	if (flags & MODEL_PREDICT_LOSS)
	{
		prediction.loss = crossEntropyLoss + l2Loss;
	}

	if (flags & MODEL_PREDICT_SCORES)
		prediction.scores = ZCache[numComputingLayers - 1];


	if (!(flags & MODEL_GRAD_UPDATE))
		return prediction;

	// Backprop
	for (int64_t layer = numComputingLayers - 1; layer >= 0; --layer)
	{
		EMatrix &W = weights[layer];
		EMatrix &dW = gradWeights[layer];
		EMatrix &db = gradBiases[layer];

		if (layer == numComputingLayers - 1)
		{
			LayerCPU::CrossEntropyLossBackward(dZCache[layer], expScoresCache, expScoresSumsCache, y);
		}
		else 
		{
			LayerCPU::ActivationLayerBackward(dZCache[layer], ZCache[layer], dACache[layer + 1], this->activation);
		}
		LayerCPU::AffineLayerBackward(dACache[layer], dW, db, ACache[layer], W, dZCache[layer], regularizationStrength);
	}

	// Update parameters
	for (unsigned i = 0; i < weights.size(); ++i)
	{
		weights[i] = weights[i].array() - gradWeights[i].array() * learnRate;
		biases[i] = biases[i].array() - gradBiases[i].array() * learnRate;
	}

	return prediction;
}