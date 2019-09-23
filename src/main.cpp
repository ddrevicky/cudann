#define EIGEN_DONT_VECTORIZE		// The GPU implementation relies on Eigen matrices not being aligned

#ifdef _WIN32
	#include <windows.h>
	#include <io.h>
	#ifdef _DEBUG
		#define _CRTDBG_MAP_ALLOC
		#include <crtdbg.h>
	#endif
#endif
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

#include "neural_net_cpu.h"
#include "neural_net_gpu.h"
#include "test_cpu.h"
#include "test_gpu.h"
#include "gpu_matrix.h"
#include "main_util.h"

using std::cout;
using std::cerr;

int main(int argc, char **argv)
{
	{
#ifdef _DEBUG
	#if defined(_WIN32)
		unsigned int crtFlags = _CRTDBG_LEAK_CHECK_DF;	// Perform automatic leak checking at program exit through a call to _CrtDumpMemoryLeaks
		crtFlags |= _CRTDBG_DELAY_FREE_MEM_DF;			// Keep freed memory blocks in the heap's linked list, assign them the _FREE_BLOCK type, and fill them with the byte value 0xDD
		crtFlags |= _CRTDBG_CHECK_ALWAYS_DF;			// Call _CrtCheckMemory at every allocation and deallocation request. _crtBreakAlloc = 323; tracks the erroneous malloc
	#endif

		// Tests
		TestCPU::RunAllTests();
		TestGPU::RunAllTests();
#endif
		HyperParameters params;
		int result = ParseArgs(argc, argv, params);
		if (result != 0)
			return result;

		// Get MNIST directory
		std::string filePath = __FILE__;
		filePath.erase(filePath.length() - strlen("/src/main.cpp"), strlen("/src/main.cpp"));
		std::string MNISTDir = filePath + "/data/mnist";

		// Make dataset
		Dataset MNIST;
		result = MakeMNIST(MNIST, MNISTDir.c_str());
		if (result != 0)
			return result;

		// Model
		float initWeightScale = 0.1f;
		int randomSeed = 256;
		unsigned inputDim = 784;
		unsigned numClasses = 10;
		std::vector<unsigned> layers = std::vector<unsigned>{ inputDim };
		layers.insert(layers.end(), params.hiddenLayers.begin(), params.hiddenLayers.end());
		layers.push_back(numClasses);

		cout << "NN Architecture: \n";
		for (int i = 0; i < layers.size(); ++i)
		{
			cout << layers[i];
			if (i < layers.size() - 1)
				cout << "-";
			else
				cout << "\n\n";
		}

		bool verboseSGD = params.verbose;			// Slows down neural net on GPU due to device to host copying

		// Train on CPU
		NeuralNetCPU nnCPU(layers, params.activation, params.regularization, initWeightScale, randomSeed, params.batchSize);
		cout << "CPU TRAINING FOR " << params.cpuEpochs << " EPOCHS\n";
		cout << "Initial test set results:";
		nnCPU.ReportAccuracy(MNIST.XVal, MNIST.yVal);
		double averageEpochTimeCPU = nnCPU.SGD(MNIST.XTrain, MNIST.yTrain, params.cpuEpochs, params.learningRate, params.decreaseLearnRateAfterEpochs, params.decreaseFactor, verboseSGD);
		cout << "Final test set results:";
		nnCPU.ReportAccuracy(MNIST.XVal, MNIST.yVal);

		// Predict on CPU
		double averagePredictTimeCPU = ProfilePredictCPU(nnCPU, MNIST.XDev, MNIST.yDev);

		// Train on GPU
		NeuralNetGPU nnGPU(layers, params.activation, params.regularization, initWeightScale, randomSeed, params.batchSize);
		cout << "\nGPU TRAINING FOR " << params.gpuEpochs << " EPOCHS\n";
		cout << "Initial test set results:";
		nnGPU.ReportAccuracy(MNIST.XVal, MNIST.yVal);
		double averageEpochTimeGPU = nnGPU.SGD(MNIST.XTrain, MNIST.yTrain, params.gpuEpochs, params.learningRate, params.decreaseLearnRateAfterEpochs, params.decreaseFactor, verboseSGD);
		cout << "Final test set results:";
		nnGPU.ReportAccuracy(MNIST.XVal, MNIST.yVal);

		// Predict on GPU
		double averagePredictTimeGPU = ProfilePredictGPU(nnGPU, MNIST.XDev, MNIST.yDev);

		// Compare performance
		std::cout << "\nPERFORMANCE COMPARISON" << std::endl;
		std::cout << "Average epoch train time: " << std::endl;
		std::cout << "CPU " << averageEpochTimeCPU << " ms" << std::endl;
		std::cout << "GPU " << averageEpochTimeGPU << " ms, " << averageEpochTimeCPU / averageEpochTimeGPU << "X speedup\n" << std::endl;
		std::cout << "Average prediction time: " << std::endl;
		std::cout << "CPU " << averagePredictTimeCPU << " ms" << std::endl;
		std::cout << "GPU " << averagePredictTimeGPU << " ms, " << averagePredictTimeCPU / averagePredictTimeGPU << "X speedup\n" << std::endl;
	}

#if defined(_WIN32) && defined(_DEBUG)
	int debugResult = _CrtDumpMemoryLeaks();
#endif

	return 0;
}