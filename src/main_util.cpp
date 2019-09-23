#include <iostream>

#include <args/args.hxx>
#include <mnist/mnist_reader.hpp>

#include "main_util.h"
#include "neural_net_cpu.h"
#include "neural_net_gpu.h"

using std::cout;
using std::cerr;

int ParseArgs(int argc, char **argv, HyperParameters &hyperParams)
{
	args::ArgumentParser parser("Trains a feed-forward NN on a CPU and GPU using the MNIST dataset.\n"
								"Example usage:\n"
								"    cudann 100 100 -b 256 -l 0.01\n"
								"    trains a network with 3 hidden layers with sizes 100, 200, 100,\n"
								"    a batch size of 256 and learning rate set to 0.01\n\n"
);
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	
	args::ValueFlag<unsigned> batchSizeFlag(parser, "uint", "Batch size", {'b', "batchSize"});
	args::ValueFlag<float> learnRateFlag(parser, "float", "Learning rate", {'l', "learnRate"});
	args::ValueFlag<unsigned> decreaseLearnRateAfterEpochsFlag(parser, "uint", "Decreases learning rate every N epochs during training", {'d', "decreaseAfter"});
	args::ValueFlag<float> decreaseFactorFlag(parser, "float", "Decreases learning rate factor", { 'f', "decreaseFactor" });
	args::ValueFlag<float> regularizationFlag(parser, "float", "L2 regularization strength", {'r', "regularization"});
	args::ValueFlag<int> cpuEpochsFlag(parser, "uint", "Number of CPU training epochs", {'c', "cpuEpochs"});
	args::ValueFlag<int> gpuEpochsFlag(parser, "uint", "Number of GPU training epochs", {'g', "gpuEpochs"});
	args::ValueFlag<std::string> activationFlag(parser, "string", "Activation function to use. Options are {relu, sigmoid} ", {'a', "activation"});
	args::Flag verboseFlag(parser, "verbose", "Display loss after each training epoch. This slows down GPU version due to device to host memory copy.", {'v', "verbose"});
	args::PositionalList<unsigned> hiddenLayersFlag(parser, "hiddenLayers", "A list of hidden layer sizes");

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        cout << parser;
        return 2;
    }
    catch (args::ParseError e)
    {
        cerr << e.what() << std::endl;
        cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        cerr << e.what() << std::endl;
        cerr << parser;
        return 1;
	}
	
	if (batchSizeFlag) 
	{ 
		unsigned int batchSize = args::get(batchSizeFlag);
		if (batchSize > 4096)
		{
			cerr << "Warning: Max allowed batch size is 4096. Option ignored.\n";
		}
		else
		{
			hyperParams.batchSize = batchSize;
		}
	}
	if (learnRateFlag) 
	{ 
		float learnRate = args::get(learnRateFlag);
		if (learnRate <= 0.0f)
		{
			cerr << "Warning: Learning rate must be a positive number. Option ignored.\n";
		}
		else
		{
			hyperParams.learningRate = learnRate;
		}
	}
	if (decreaseLearnRateAfterEpochsFlag)
	{
		unsigned int decreaseLearnRateAfterEpochs = args::get(decreaseLearnRateAfterEpochsFlag);
		if (decreaseLearnRateAfterEpochs == 0)
		{
			cerr << "Warning: Half learn rate after epochs must not be zero. Option ignored.\n";
		}
		else
		{
			hyperParams.decreaseLearnRateAfterEpochs = decreaseLearnRateAfterEpochs;
		}
	}
	if (decreaseFactorFlag)
	{
		float decreaseLearnRateFactor = args::get(decreaseFactorFlag);
		if (decreaseLearnRateFactor <= 0.0f)
		{
			cerr << "Warning: Decrease learn rate factor must be a positive number. Option ignored.\n";
		}
		else
		{
			hyperParams.decreaseFactor = decreaseLearnRateFactor;
		}
	}
	if (regularizationFlag) 
	{ 
		float regularization = args::get(regularizationFlag);
		if (regularization < 0.0f)
		{
			cerr << "Warning: Regularization must be a non-negative number. Option ignored.\n";
		}
		else
		{
			hyperParams.regularization = regularization;
		}
	}
	hyperParams.verbose = verboseFlag;
	if (cpuEpochsFlag) 
	{ 
		hyperParams.cpuEpochs = args::get(cpuEpochsFlag);
	}
	if (gpuEpochsFlag) 
	{ 
		hyperParams.gpuEpochs = args::get(gpuEpochsFlag);
	}
	if (hiddenLayersFlag) 
	{ 
		const std::vector<unsigned> hiddenLayers(args::get(hiddenLayersFlag));
		hyperParams.hiddenLayers = hiddenLayers;
	}
	if (activationFlag)
	{
		const std::string activation = args::get(activationFlag);
		if (activation == "relu")
			hyperParams.activation = ActivationFunction::ReLU;
		else if (activation == "sigmoid")
			hyperParams.activation = ActivationFunction::Sigmoid;
		else
		{
			cerr << "Warning: Unknown activation function. Option ignored.\n";
		}
	}

	// Print chosen parameters
	cout << "\nHYPERPARAMETERS (to see how to change these run ./cudann --help)\n";
	cout << "Batch size " << hyperParams.batchSize << "\n";
	cout << "Learning rate " << hyperParams.learningRate << "\n";
	cout << "Decrease learning rate after " << hyperParams.decreaseLearnRateAfterEpochs << " epochs\n";
	cout << "Learning rate decrease factor " << hyperParams.decreaseFactor << "\n";
	cout << "Regularization strength " << hyperParams.regularization << "\n";
	cout << "Hidden layers: ";
	for (int i = 0; i < hyperParams.hiddenLayers.size(); ++i)
		cout << hyperParams.hiddenLayers[i] << " ";
	cout << "\n";
	cout << "Activation function: " << (hyperParams.activation == ReLU ? "ReLU" : "Sigmoid") << " \n";
	cout << "CPU train epochs " << hyperParams.cpuEpochs << "\n";
	cout << "GPU train epochs " << hyperParams.gpuEpochs << "\n";
	cout << std::boolalpha;   
	cout << "Verbose training " << hyperParams.verbose << "\n";
	if (hyperParams.verbose)
	{
		cout << "WARNING: Verbose training increases training time for the GPU version.\n";
		cout << "Disable it for profiling please.\n";
	}
	cout << "\n";
    return 0;
}

int MakeMNIST(Dataset &MNIST, const char *dir)
{
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> loaded_dataset =
		mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dir);
	
	if (loaded_dataset.training_images.size() == 0)
	{
		cerr << "Dataset could not be loaded. Please run the program from the directory where it is located.\n";
		return 1;
	}

	int nTrain = 50000;
	int nVal = 10000;
	int nTest = 10000;
	int nDev = 2048;
	int nDimensions = 784;

	MNIST.XTrain = EMatrix(nTrain, nDimensions);
	MNIST.yTrain = EMatrix(nTrain, 1);
	MNIST.XVal = EMatrix(nVal, nDimensions);
	MNIST.yVal = EMatrix(nVal, 1);
	MNIST.XTest = EMatrix(nTest, nDimensions);
	MNIST.yTest = EMatrix(nTest, 1);
	MNIST.XDev = EMatrix(nDev, nDimensions);
	MNIST.yDev = EMatrix(nDev, 1);

	// Dev
	for (int r = 0; r < nDev; ++r)
	{
		MNIST.yDev(r, 0) = loaded_dataset.training_labels[r];
		for (int c = 0; c < nDimensions; ++c)
		{
			MNIST.XDev(r, c) = loaded_dataset.training_images[r][c];
		}
	}
	// Train
	for (int r = 0; r < nTrain; ++r)
	{
		MNIST.yTrain(r, 0) = loaded_dataset.training_labels[r];
		for (int c = 0; c < nDimensions; ++c)
		{
			MNIST.XTrain(r, c) = loaded_dataset.training_images[r][c];
		}
	}

	// Val
	for (int r = 0; r < nVal; ++r)
	{
		MNIST.yVal(r, 0) = loaded_dataset.training_labels[nTrain + r];
		for (int c = 0; c < nDimensions; ++c)
		{
			MNIST.XVal(r, c) = loaded_dataset.training_images[nTrain + r][c];
		}
	}

	// Test
	for (int r = 0; r < nTest; ++r)
	{
		MNIST.yTest(r, 0) = loaded_dataset.test_labels[r];
		for (int c = 0; c < nDimensions; ++c)
		{
			MNIST.XTest(r, c) = loaded_dataset.test_images[r][c];
		}
	}

	return 0;
}

double ProfilePredictCPU(NeuralNetCPU &net, EMatrix &X, EMatrix &y)
{
	unsigned flags = MODEL_PREDICT_SCORES;
	int numRuns = 40;
	CPUTimer timerCPU;
	timerCPU.Start();
	for (int i = 0; i < numRuns; ++i)
	{
		net.Loss(X, y, flags, 0.0f);
	}
	timerCPU.Stop();
	double averageTime = timerCPU.GetElapsedTime() / numRuns;
	return averageTime;
}

double ProfilePredictGPU(NeuralNetGPU &net, EMatrix &X, EMatrix &y)
{
	E_TO_GPU_MATRIX(X, X_g);
	E_TO_GPU_MATRIX(y, y_g);
	
	unsigned flags = MODEL_PREDICT_SCORES;
	int numRuns = 40;
	CPUTimer timerCPU;
	timerCPU.Start();
	for (int i = 0; i < numRuns; ++i)
	{
		net.Loss(&X_g, &y_g, flags, 0.0f);
	}
	timerCPU.Stop();
	double averageTime = timerCPU.GetElapsedTime() / numRuns;
	return averageTime;

	X_g.Release();
	y_g.Release();
	return averageTime;
}