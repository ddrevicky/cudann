#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "neural_net_gpu.h"
#include "layers_gpu.h"
#include "gpu_matrix.h"
#include "gpu_scalar.h"
#include "cuda_utility.h"
#include "e_matrix.h"

GPUMemoryPool::GPUMemoryPool()
{
}

GPUMemoryPool::GPUMemoryPool(size_t poolSizeInBatches, int fullBatchRows, int XCols, int yCols)
{
	size_t XElementsPerBatch = fullBatchRows * XCols;
	size_t yElementsPerBatch = fullBatchRows * yCols;
	size_t poolSizeInBytes = poolSizeInBatches * (XElementsPerBatch + yElementsPerBatch) * sizeof(float);

	// Pinned host memory required for async copy
	cudaErrCheck(cudaMallocHost(&hp_reserved, poolSizeInBytes));
	memset(hp_reserved, 0, poolSizeInBytes);
	hp_XStart = hp_reserved;
	hp_yStart = hp_reserved + XElementsPerBatch * poolSizeInBatches;

	maxBatches = int(poolSizeInBatches);
	actualBatches = 0;
	bytesReserved = poolSizeInBytes;
	X = new GPUMatrix[poolSizeInBatches];
	y = new GPUMatrix[poolSizeInBatches];
	cudaErrCheck(cudaMallocManaged(&d_reserved, poolSizeInBytes));
	cudaErrCheck(cudaMemset(d_reserved, 0, poolSizeInBytes));

	// Memory Pool is arranged like this: X1X2...Xny1y2...yn
	float *d_X = d_reserved;
	float *d_y = d_reserved + XElementsPerBatch * poolSizeInBatches;
	for (int batch = 0; batch < poolSizeInBatches; ++batch)
	{
		X[batch].cols = XCols;
		X[batch].realElementCount = fullBatchRows * XCols;
		X[batch].data = d_X;
		d_X += XElementsPerBatch;
		
		y[batch].cols = yCols;
		y[batch].realElementCount = fullBatchRows * yCols;
		y[batch].data = d_y;
		d_y += yElementsPerBatch;
	}

	cudaErrCheck(cudaStreamCreate(&asyncCopyStream));
	cudaErrCheck(cudaEventCreate(&asyncCopyFinishedEvent));
}

void GPUMemoryPool::Release()
{
	delete[] X; X = nullptr;
	delete[] y; y = nullptr;
	cudaErrCheck(cudaFreeHost(hp_reserved)); hp_reserved = nullptr;
	cudaErrCheck(cudaFree(d_reserved)); d_reserved = nullptr;
	cudaErrCheck(cudaStreamDestroy(asyncCopyStream));
	cudaErrCheck(cudaEventDestroy(asyncCopyFinishedEvent));
}

NeuralNetGPU::NeuralNetGPU()
{
}

NeuralNetGPU::NeuralNetGPU(std::vector<unsigned> layerSizes, ActivationFunction activation, float regStrength, float weightScale, int randSeed, int batchSize)
{
	UAssert(layerSizes.size() > 1, "Number of layers must be greater than 1.");
	UAssert(regStrength >= 0, "Regularization strength must be non-negative.");
	this->layerSizes = layerSizes;
	this->regularizationStrength = regStrength;
	this->fullBatchRows = batchSize;
	this->currentCacheRows = batchSize;
	this->activation = activation;

	std::srand(randSeed);
	std::default_random_engine generator(randSeed);
	std::normal_distribution<float> standardNormal(0.0f, 1.0f);

	// Init weights, biases and preallocate matrices for gradients
	size_t nComputingLayers = layerSizes.size() - 1;
	for (size_t layer = 0; layer < nComputingLayers; ++layer)
	{
		// W
		EMatrix weightCPU = EMatrix(layerSizes[layer], layerSizes[layer + 1]);
		for (int r = 0; r < weightCPU.rows(); ++r)
		{
			for (int c = 0; c < weightCPU.cols(); ++c)
			{
				weightCPU(r, c) = weightScale * standardNormal(generator) / sqrtf(layerSizes[layer]);
			}
		}
		GPUMatrix weight = GPUMatrix(layerSizes[layer], layerSizes[layer + 1]);
		weight.SetFromMem(weightCPU.data());
		weights.push_back(weight);

		// dW
		GPUMatrix gradWeight = GPUMatrix(weight.rows, weight.cols);
		gradWeights.push_back(gradWeight);

		// b
		GPUMatrix bias = GPUMatrix(1, layerSizes[layer + 1], GPU_SET_ZERO);
		biases.push_back(bias);

		// db
		GPUMatrix gradBias = GPUMatrix(bias.rows, bias.cols);
		gradBiases.push_back(gradBias);
	}

	// Loss
	crossEntropyLoss = GPUScalar(GPU_SET_ZERO);
	l2Loss = GPUScalar(GPU_SET_ZERO);

	// Cache
	int maxMatrixElementCount = 0;
	int maxWeightRows = 0;
	for (size_t layer = 0; layer < nComputingLayers; ++layer)
	{
		ACache.push_back(GPUMatrix(fullBatchRows, layerSizes[layer]));
		dACache.push_back(GPUMatrix(fullBatchRows, layerSizes[layer]));
		ZCache.push_back(GPUMatrix(fullBatchRows, layerSizes[layer + 1]));
		dZCache.push_back(GPUMatrix(fullBatchRows, layerSizes[layer + 1]));

		// Find the largest matrix by element count
		maxMatrixElementCount = std::max(maxMatrixElementCount, int(fullBatchRows * layerSizes[layer]));
		maxMatrixElementCount = std::max(maxMatrixElementCount, int(fullBatchRows * layerSizes[layer + 1]));
		maxMatrixElementCount = std::max(maxMatrixElementCount, weights[layer].realElementCount);

		// Find the weight with maximum rows
		maxWeightRows = std::max(maxWeightRows, weights[layer].rows);
	}

	intermediate = GPUMatrix(maxMatrixElementCount, 1);			// Utility matrix for intermediate results
	l2IntermediateVec = GPUMatrix(maxWeightRows, 1);			// Utility vector for L2 regularization intermediate results
	maxScores = GPUMatrix(fullBatchRows, 1);
	expScores = GPUMatrix(fullBatchRows, layerSizes[nComputingLayers]);
	expScoresSums = GPUMatrix(fullBatchRows, 1);


	// GPU and pinned host staging area for multiple batches. Contiguous piece of memory containing both X and y.
	// There is a single host staging area to which data is copied before initianing async mem transfer to the GPU (from this host area).
	// There are two multibatch GPU memory pools which are ping-ponged between. One is initialized, contains multiple batches and is 
	// used for actual computation. At that time the other pool is being asynchronously copied to. 
	int XCols = layerSizes[0];
	int yCols = 1;

	// GPU memory. Use 2 pools and ping-pong between them
	memPools[0] = GPUMemoryPool(poolSizeInBatches, fullBatchRows, XCols, yCols);
	memPools[1] = GPUMemoryPool(poolSizeInBatches, fullBatchRows, XCols, yCols);

	// Create compute stream
	cudaErrCheck(cudaStreamCreate(&computeStream));
}

void NeuralNetGPU::ReportAccuracy(EMatrix &X, EMatrix &y)
{
	ReshapeCache(fullBatchRows);

	int batchSize = fullBatchRows;
	int nBatches = int(ceil(float(X.rows()) / float(batchSize)));
	float totalLoss = 0.0f;
	int correctPredictions = 0;

	GPUMatrix XBatch = GPUMatrix(batchSize, int(X.cols()));
	GPUMatrix yBatch = GPUMatrix(batchSize, int(y.cols()));
	for (int i = 0; i < nBatches; ++i)
	{
		int startRow = i * batchSize;
		int endRow = std::min(startRow + batchSize, int(X.rows() - 1));
		int batchActualSize = endRow - startRow;

		if (batchActualSize < currentCacheRows)
		{
			ReshapeCache(batchActualSize);
			XBatch.rows = batchActualSize;
			yBatch.rows = batchActualSize;
		}

		XBatch.SetFromMem(X.data() + startRow * int(X.cols()));
		yBatch.SetFromMem(y.data() + startRow * int(y.cols()));

		unsigned int flags = MODEL_PREDICT_LOSS | MODEL_PREDICT_SCORES;
		float learnRate = 0.0f;
		Prediction batchPrediction = Loss(&XBatch, &yBatch, flags, learnRate);

		// Accumulate weighted batch loss
		totalLoss += batchPrediction.loss * batchActualSize;

		// Accumulate correct predictions
		EMatrix yBatchCPU = y.block(startRow, 0, batchActualSize, y.cols());
		for (int r = 0; r < yBatchCPU.rows(); ++r)
		{
			int maxRow, maxCol;
			batchPrediction.scores.row(r).maxCoeff(&maxRow, &maxCol);

			int predictedClass = maxCol;
			int correctClass = int(yBatchCPU(r, 0));

			if (predictedClass == correctClass)
				++correctPredictions;
		}
	}
	XBatch.Release();
	yBatch.Release();

	float accuracy = float(correctPredictions) / float(y.rows());
	totalLoss /= X.rows();
	std::cout << "Loss " << totalLoss << ", Accuracy " << 100 * accuracy << "%\n";
}

// Ping-pong between 2 memory pools on the GPU. While one pool is used for computation the other pool can receive asynchronous
// memory copy of the next series of batches. We copy multiple batches at once to reduce the transfer overhead.
void NeuralNetGPU::TrainEpoch(EMatrix &XCPU, EMatrix &yCPU, unsigned int flags, float learnRate)
{
	int nBatchesTotal = (int(XCPU.rows()) + fullBatchRows - 1) / fullBatchRows;
	int nBatchesSentToGPU = 0;
	int nBatchesProcessed= 0;
	bool firstBatch = true;

	int copyingPoolIdx = 0;
	int computingPoolIdx = 1;

	memPools[0].actualBatches = 0;
	memPools[1].actualBatches = 0;
	int poolBatch = 0;

	bool computingPoolExhausted = true;
	while (nBatchesProcessed != nBatchesTotal)
	{
		// Copy to GPU
		bool allBatchesSentToGPU = nBatchesSentToGPU == nBatchesTotal;
		if (computingPoolExhausted && !allBatchesSentToGPU)
		{
			int nBatchesRemainingInEpoch = nBatchesTotal - nBatchesSentToGPU;
			int nBatchesToCopy = std::min(poolSizeInBatches, nBatchesRemainingInEpoch);

			int startRow = nBatchesSentToGPU * fullBatchRows;
			int onePastEndRow = std::min(startRow + nBatchesToCopy * fullBatchRows, int(XCPU.rows()));
			int nRowsToCopy = onePastEndRow - startRow;
			assert(nRowsToCopy > 0);

			GPUMemoryPool *copyingPool = &memPools[copyingPoolIdx];

			// Copy from XCPU, yCPU to host pinned. Eigen vectorization must be off so that no alignment is used
			cudaErrCheck(cudaStreamSynchronize(copyingPool->asyncCopyStream));		// Sync in case the async stream is still reading from the pinned memory
			memcpy(copyingPool->hp_XStart, (void*)(XCPU.data() + startRow * XCPU.cols()), nRowsToCopy * XCPU.cols() * sizeof(float));
			memcpy(copyingPool->hp_yStart, (void*)(yCPU.data() + startRow * yCPU.cols()), nRowsToCopy * yCPU.cols() * sizeof(float));

			// Copy from host pinned to GPU
			// We might copy some invalid data from host pinned memory for the last batch, but it will not be used
			if (firstBatch)
			{
				cudaErrCheck(cudaMemcpy(copyingPool->d_reserved, copyingPool->hp_reserved, copyingPool->bytesReserved, cudaMemcpyHostToDevice));
			}
			else
			{
				cudaErrCheck(cudaMemcpyAsync(copyingPool->d_reserved, copyingPool->hp_reserved, copyingPool->bytesReserved, cudaMemcpyHostToDevice, copyingPool->asyncCopyStream));
				cudaErrCheck(cudaEventRecord(copyingPool->asyncCopyFinishedEvent, copyingPool->asyncCopyStream));
			}

			// Update convenience matrices row counts
			copyingPool->actualBatches = nBatchesToCopy;
			for (int copiedBatch = 0; copiedBatch < copyingPool->actualBatches; ++copiedBatch)
			{
				int actualRows = fullBatchRows;
				bool isLastBatch = copiedBatch == nBatchesToCopy - 1;
				bool isSmallerThanFull = nRowsToCopy % fullBatchRows != 0;
				if (isLastBatch && isSmallerThanFull)
				{
					actualRows = nRowsToCopy % fullBatchRows;	// Last batch for the epoch might have fewer rows than a full batch
				}
				copyingPool->X[copiedBatch].rows = actualRows;
				copyingPool->y[copiedBatch].rows = actualRows;
			}

			nBatchesSentToGPU += nBatchesToCopy;
			computingPoolExhausted = false;
		}

		// Perform forward and backward pass
		if (firstBatch)
		{
			firstBatch = false;
			std::swap(computingPoolIdx, copyingPoolIdx);
			computingPoolExhausted = true;
		}
		else
		{
			GPUMemoryPool *computingPool = &memPools[computingPoolIdx];
			assert(poolBatch < computingPool->actualBatches);

			GPUMatrix *X = &computingPool->X[poolBatch];
			GPUMatrix *y = &computingPool->y[poolBatch];

			if (X->rows != currentCacheRows)
				ReshapeCache(X->rows);

			Prediction prediction = Loss(X, y, flags, learnRate);
			if ((flags & MODEL_PREDICT_LOSS) && nBatchesProcessed == 0)
				std::cout << "Loss " << prediction.loss;
			
			++nBatchesProcessed;
			++poolBatch;

			computingPoolExhausted = poolBatch == computingPool->actualBatches;
			if (computingPoolExhausted)
			{
				// Synchronize the default stream with the async copy stream so that in the next iteration it starts computing
				// on completely copied data
				std::swap(computingPoolIdx, copyingPoolIdx);
				cudaErrCheck(cudaStreamWaitEvent(0, computingPool->asyncCopyFinishedEvent, 0));
				poolBatch = 0;
			}
		}
	}
}

double NeuralNetGPU::SGD(EMatrix &XTrain, EMatrix &yTrain, int epochs, float learnRate, int decreaseLearnRateAfterEpochs, float decreaseFactor, bool verbose)
{
	double totalEpochTime = 0.0;
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
		EMatrix XShuffled = permutation * XTrain;
		EMatrix yShuffled = permutation * yTrain;

		unsigned int modelFlags = MODEL_GRAD_UPDATE;
		if (verbose)
			modelFlags |= MODEL_PREDICT_LOSS;
		CPUTimer epochTimer;
		epochTimer.Start();
		TrainEpoch(XShuffled, yShuffled, modelFlags, learnRate);
		epochTimer.Stop();
		totalEpochTime += epochTimer.GetElapsedTime();
		std::cout << std::endl;
	}
	return totalEpochTime / double(epochs);
}

Prediction NeuralNetGPU::Loss(GPUMatrix *X, GPUMatrix *y, unsigned int flags, float learnRate)
{
	size_t nComputingLayers = layerSizes.size() - 1;
	GPUMatrix *prevLayerA = X;
	for (size_t layer = 0; layer < nComputingLayers; ++layer)
	{
		GPUMatrix &W = weights[layer];
		GPUMatrix &b = biases[layer];
		GPUMatrix &Z = ZCache[layer];

		LayerGPU::AffineLayerForward(*prevLayerA, W, b, Z, computeStream);
		if (layer < nComputingLayers - 1)
		{
			LayerGPU::ActivationLayerForward(Z, ACache[layer + 1], activation, computeStream);
			prevLayerA = &ACache[layer + 1];
		}
		else
		{
			LayerGPU::CrossEntropyLossForward(Z, *y, maxScores, expScores, expScoresSums, crossEntropyLoss.data, computeStream);
			LayerGPU::L2RegularizedLoss(weights, l2Loss.data, regularizationStrength, intermediate, l2IntermediateVec, computeStream);
		}
	}

	Prediction prediction;
	if (flags & MODEL_PREDICT_LOSS)
	{
		cudaErrCheck(cudaDeviceSynchronize());
		prediction.loss = *(crossEntropyLoss.data) + *(l2Loss.data);
	}

	if (flags & MODEL_PREDICT_SCORES)
	{
		cudaErrCheck(cudaDeviceSynchronize());
		GPUMatrix &scores = ZCache[nComputingLayers - 1];
		Eigen::Map<EMatrix> map(scores.data, scores.rows, scores.cols);
		prediction.scores = EMatrix(map);
	}

	if (!(flags & MODEL_GRAD_UPDATE))
		return prediction;

	// Backprop
	for (int64_t layer = nComputingLayers - 1; layer >= 0; --layer)
	{
		GPUMatrix &W = weights[layer];
		GPUMatrix &dW = gradWeights[layer];
		GPUMatrix &db = gradBiases[layer];

		if (layer == nComputingLayers - 1)
		{
			LayerGPU::CrossEntropyLossBackward(expScores, expScoresSums, *y, dZCache[layer], computeStream);
		}
		else
		{
			LayerGPU::ActivationLayerBackward(dZCache[layer], ZCache[layer], dACache[layer + 1], activation, computeStream);
		}
		GPUMatrix &WTrans = intermediate;
		GPUMatrix &ATrans = intermediate;
		GPUMatrix &A = layer > 0 ? ACache[layer] : *X;
		LayerGPU::AffineLayerBackward(dACache[layer], dW, db, A, ATrans, W, WTrans, dZCache[layer], regularizationStrength, computeStream);
	}

	// Update parameters using gradient
	for (size_t i = 0; i < weights.size(); ++i)
	{
		OpGPU::MatMatAdd(weights[i], gradWeights[i], weights[i], -learnRate, computeStream);
	}

	return prediction;
}

void NeuralNetGPU::ReshapeCache(int newRows)
{
	currentCacheRows = newRows;

	size_t nComputingLayers = layerSizes.size() - 1;
	for (size_t layer = 0; layer < nComputingLayers; ++layer)
	{
		ACache[layer].rows = newRows;
		dACache[layer].rows = newRows;
		ZCache[layer].rows = newRows;
		dZCache[layer].rows = newRows;
	}
	maxScores.rows = newRows;
	expScores.rows = newRows;
	expScoresSums.rows = newRows;
}

NeuralNetGPU::~NeuralNetGPU() 
{
	// Release Cache
	for (GPUMatrix &x : ZCache) x.Release();
	ZCache.clear();
	for (GPUMatrix &x : ACache) x.Release();
	ACache.clear();
	for (GPUMatrix &x : dACache) x.Release();
	dACache.clear();
	for (GPUMatrix &x : dZCache) x.Release();
	dZCache.clear();

	maxScores.Release();
	expScores.Release();
	expScoresSums.Release();
	intermediate.Release();
	l2IntermediateVec.Release();

	crossEntropyLoss.Release();
	l2Loss.Release();

	// Release params and gradients
	for (GPUMatrix &x : weights) x.Release();
	weights.clear();
	for (GPUMatrix &x : biases) x.Release();
	biases.clear();
	for (GPUMatrix &x : gradWeights) x.Release();
	gradWeights.clear();
	for (GPUMatrix &x : gradBiases) x.Release();
	gradBiases.clear();

	memPools[0].Release();
	memPools[1].Release();

	cudaErrCheck(cudaStreamDestroy(computeStream));
}