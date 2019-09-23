#pragma once

#include <vector>
#include <tuple>
#include <map>
#include <string>

#include "model.h"
#include "layers_cpu.h"
#include "utility.h"
#include "e_matrix.h"
#include "activation_functions.h"

struct Prediction;

class NeuralNetCPU
{
public:
	NeuralNetCPU();
	NeuralNetCPU(std::vector<unsigned> sizes, ActivationFunction activation, float regStrength, float weightScale, int randSeed, int batchSize);
	double SGD(EMatrix &XTrain, EMatrix &yTrain, int epochs, float learnRate, int decreaseLearnRateAfterEpochs, float decreaseFactor, bool verbose);
	void ReportAccuracy(EMatrix &x, EMatrix &y);
	Prediction Loss(EMatrix &xTrain, EMatrix &yTrain, unsigned int flags, float learnRate);

public:
	std::vector<unsigned> sizes;
	std::vector<EMatrix> weights;
	std::vector<EMatrix> biases;
	std::vector<EMatrix> gradWeights;
	std::vector<EMatrix> gradBiases;

private:
	ActivationFunction activation;
	float regularizationStrength = 0.0f;
	std::vector<EMatrix> ZCache;
	std::vector<EMatrix> ACache;
	std::vector<EMatrix> dACache;
	std::vector<EMatrix> dZCache;
	EMatrix expScoresCache;
	EMatrix expScoresSumsCache;
	int batchSize;
};