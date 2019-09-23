#pragma once

#include "model.h"
#include "utility.h"
#include "e_matrix.h"
#include "gpu_matrix.h"
#include "gpu_scalar.h"
#include "activation_functions.h"
#include <cuda_runtime.h>

struct Prediction;

struct GPUMemoryPool
{
	GPUMemoryPool();
	GPUMemoryPool(size_t poolSizeInBatches, int fullBatchRows, int XCols, int yCols);
	void Release();

	float *hp_reserved = nullptr;
	float *hp_XStart = nullptr;
	float *hp_yStart = nullptr;
	float *d_reserved = nullptr;
	size_t bytesReserved = 0;
	int maxBatches = 0;
	int actualBatches = 0;
	GPUMatrix *X = nullptr;
	GPUMatrix *y = nullptr;

	cudaStream_t asyncCopyStream = 0;
	cudaEvent_t asyncCopyFinishedEvent = 0;
};

class NeuralNetGPU
{
public:
	NeuralNetGPU();
	NeuralNetGPU(std::vector<unsigned> sizes, ActivationFunction activation, float regStrength, float weightScale, int randSeed, int batchSize);
	~NeuralNetGPU();
	double SGD(EMatrix &XTrain, EMatrix &yTrain, int epochs, float learnRate, int decreaseLearnRateAfterEpochs, float decreaseFactor, bool verbose);
	void ReportAccuracy(EMatrix &x, EMatrix &y);
	Prediction Loss(GPUMatrix *XBatch, GPUMatrix *yBatch, unsigned int flags, float learnRate);

private:
	void TrainEpoch(EMatrix &XCPU, EMatrix &yCPU, unsigned int flags, float learnRate);
	void ReshapeCache(int newRows);

public:
	std::vector<unsigned> layerSizes;

	std::vector<GPUMatrix> weights;
	std::vector<GPUMatrix> biases;
	std::vector<GPUMatrix> gradWeights;
	std::vector<GPUMatrix> gradBiases;

private:
	ActivationFunction activation;
	float regularizationStrength = 0.0f;
	int fullBatchRows = 0;
	int currentCacheRows = 0;

	int computingBatch = 0;
	int poolSizeInBatches = 10;
	GPUMemoryPool memPools[2];
	cudaStream_t computeStream = 0;

	GPUMatrix maxScores;
	GPUMatrix expScores;
	GPUMatrix expScoresSums;
	GPUMatrix intermediate;
	GPUMatrix l2IntermediateVec;
	std::vector<GPUMatrix> ZCache;
	std::vector<GPUMatrix> ACache;
	std::vector<GPUMatrix> dACache;
	std::vector<GPUMatrix> dZCache;

	GPUScalar crossEntropyLoss;
	GPUScalar l2Loss;
};