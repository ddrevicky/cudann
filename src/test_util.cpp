#include <iostream>

#include "test_util.h"
#include "e_matrix.h"
#include "gpu_matrix.h"

void TestUtil::RandomInit(EMatrix &m)
{
	std::srand(TEST_RANDOM_SEED);
	for (int r = 0; r < int(m.rows()); ++r)
	{
		for (int c = 0; c < int(m.cols()); ++c)
		{
			m(r, c) = 10.0f * float(std::rand()) / float(RAND_MAX);
		}
	}
}

static void PrintLargestDifference(EMatrix &e1, EMatrix &e2)
{
	float largestDifference = 0.0f;
	int rowLargest = 0;
	int colLargest = 0;
	float totalDifference = 0.0f;
	for (long r = 0; r < e1.rows(); ++r)
	{
		for (long c = 0; c < e1.cols(); ++c)
		{
			float difference = fabs(e1(r, c) - e2(r, c));
			totalDifference += difference;
			if (difference > largestDifference)
			{
				largestDifference = difference;
				rowLargest = r;
				colLargest = c;
			}
		}
	}

	std::cout << "Total difference: " << totalDifference << std::endl;
	std::cout << "Largest difference: " << largestDifference << "row " << rowLargest << " col " << colLargest << std::endl;
	std::cout << "Value1: " << e1(rowLargest, colLargest) << "\nValue2: " << e2(rowLargest, colLargest) << std::endl;
}

static void PrintLargestDifference(EMatrix &e, GPUMatrix &g)
{
	Eigen::Map<EMatrix> gMap(g.data, g.rows, g.cols);
	float largestDifference = 0.0f;
	int rowLargest = 0;
	int colLargest = 0;
	float totalDifference = 0.0f;
	for (long r = 0; r < e.rows(); ++r)
	{
		for (long c = 0; c < e.cols(); ++c)
		{
			float difference = fabs(e(r, c) - g.data[r * g.cols + c]);
			totalDifference += difference;
			if (difference > largestDifference)
			{
				largestDifference = difference;
				rowLargest = r;
				colLargest = c;
			}
		}
	}
	std::cout << "Total difference: " << totalDifference << std::endl;
	std::cout << "Largest difference: " << largestDifference << "row " << rowLargest << " col " << colLargest << std::endl;
}

bool TestUtil::IsApprox(EMatrix &e, GPUMatrix &g)
{
	Eigen::Map<EMatrix> gMap(g.data, g.rows, g.cols);
	return e.isApprox(gMap);
}

bool TestUtil::AlmostEqual(double x, double y, double tolerance)
{
	double diff = abs(x - y);
	double mag = std::max(abs(x), abs(y));
	if (mag > tolerance)
		return diff / mag <= tolerance;
	else
		return diff <= tolerance;
}

double TestUtil::RelError(double x, double y, double tolerance)
{
	double diff = abs(x - y);
	double mag = std::max(abs(x), abs(y));
	double result;
	if (mag > tolerance)
		result = diff / mag;
	else
		result = diff;
	return result;
}

double TestUtil::RelError(EMatrix &e1, EMatrix &e2, double tolerance)
{
	assert(e1.cols() == e2.cols() && e1.rows() == e2.rows());

	double maxError = 0.0f;
	for (int r = 0; r < e1.rows(); ++r)
	{
		for (int c = 0; c < e1.cols(); ++c)
		{
			double error = RelError(e1(r, c), e2(r, c));
			maxError = std::max(maxError, error);
		}
	}
	return maxError;
}
