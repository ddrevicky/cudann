#include <vector>
#include "e_matrix.h"
#include "activation_functions.h"
#include "neural_net_cpu.h"
#include "neural_net_gpu.h"

struct Dataset
{
	EMatrix XTrain;
	EMatrix yTrain;
	EMatrix XVal;
	EMatrix yVal;
	EMatrix XTest;
	EMatrix yTest;
	EMatrix XDev;
	EMatrix yDev;
};

struct HyperParameters
{
	unsigned batchSize = 256;
	float regularization = 0.00001f;
	float learningRate = 0.01f;
	unsigned decreaseLearnRateAfterEpochs = 20;
	float decreaseFactor = 0.1f;
	unsigned cpuEpochs = 3;
	unsigned gpuEpochs = 30;
	std::vector<unsigned> hiddenLayers = std::vector<unsigned>{ 300, 300 };
	bool verbose = false;
	ActivationFunction activation = ActivationFunction::ReLU;
};

int MakeMNIST(Dataset &MNIST, const char *dir);
double ProfilePredictCPU(NeuralNetCPU &net, EMatrix &X, EMatrix &y);
double ProfilePredictGPU(NeuralNetGPU &net, EMatrix &X, EMatrix &y);
int ParseArgs(int argc, char **argv, HyperParameters &hyperParams);