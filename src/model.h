#pragma once

#include "e_matrix.h"

// Binary flags
#define MODEL_PREDICT_LOSS		1 << 0
#define MODEL_PREDICT_SCORES	1 << 1
#define MODEL_GRAD_UPDATE		1 << 2

struct Prediction
{
	float loss = -1.0f;
	EMatrix scores;
};