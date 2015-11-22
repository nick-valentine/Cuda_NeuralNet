#include "NeuralNetKernel.h"

float * g_weights_d;

/*

suppose 4 inputs, 3 per layer, 3 layers, 2 outputs
	o - o - o - o - o
	  x   x   x   x
	o - o - o - o - o
	  x   x   x
	o - o - o - o
	  x 
	o -

	take layer n, each o has |n-1| weights
	| input layer | x | layer one | = num weights in first layer
	| n - 1 | x | n | = weights in each layer after

	for this one:
	4 x 3 = 12
	3 x 3 = 9
	3 x 3 = 9
	3 x 2 = 6

	12 + 9 + 9 + 6 = 36 weights total

	0  []   12 []   21 []   30 []
	1  []   13 []   22 []   31 []
	2  []   14 []   23 []   32 []
	3  []

	4  []   15 []   24 []   33 []
	5  []   16 []   25 []   34 []
	6  []   17 []   26 []   35 []
	7  []

	8  []   18 []   27 []
	9  []   19 []   28 []
	10 []   20 []   29 []
	11 []

	given num_per_layer
		  num_in_input_layer
		  num_in_output_layer
		  num_weights
	
*/

void start_cuda(int size, float * weights)
{
	cudaMalloc((void **)&g_weights_d, size * sizeof(float));

	cudaMemcpy(g_weights_d, weights, size * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_weights(int size, float * weights)
{
	cudaMemcpy(g_weights_d, weights, size * sizeof(float), cudaMemcpyHostToDevice);
}

void print_weights(int size)
{
	float *weights = (float *)malloc(size * sizeof(float));
	cudaMemcpy(weights, g_weights_d, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; ++i)
	{
		printf("%f\n", weights[i]);
	}
	free(weights);
}

void end_cuda()
{
	cudaFree(g_weights_d);
}

void call_cuda_neural_net(int num_per_sweeper, int num_per_layer, int num_per_input, int num_per_output, int num_sweepers, int num_weights, int num_layers, float response, float *inputs, float * outputs)
{
	float /**Weights_d,*/ *inputs_d, *outputs_d;

	//cudaMalloc((void **)&Weights_d, num_weights * num_sweepers * sizeof(float));
	cudaMalloc((void **)&inputs_d, num_per_input * num_sweepers * sizeof(float));
	cudaMalloc((void **)&outputs_d, num_per_output * num_sweepers * sizeof(float));

	//cudaMemcpy(Weights_d, weights, num_weights * num_sweepers * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(inputs_d, inputs, num_per_input * num_sweepers * sizeof(float), cudaMemcpyHostToDevice);

	//inputs do not count since they do not require processing, more inputs only mean the weights vectors of the first layer are longer
	int max_neurons_per_layer = std::max(num_per_output, num_per_layer);

	dim3 blocks(num_sweepers, 1, 1);
	dim3 threads(max_neurons_per_layer, 1, 1);

	cuda_neural_net <<< blocks, threads, max_neurons_per_layer * sizeof(float)>>>
			(g_weights_d, num_per_sweeper, num_per_layer, 
			num_per_input, num_per_output, num_weights, 
			num_layers, response, inputs_d, outputs_d);

	cudaMemcpy(outputs, outputs_d, num_per_output * num_sweepers * sizeof(float), cudaMemcpyDeviceToHost);

	//cudaFree(Weights_d);
	cudaFree(inputs_d);
	cudaFree(outputs_d);
}

__global__ void cuda_neural_net(float *Weights_D, int num_per_sweeper, int num_per_layer, int num_per_input, int num_per_output, int num_weights, int num_layers, float response, float *inputs_d, float *outputs_d)
{

	extern __shared__ float buffer[];

	int start_of_weights = blockIdx.x * num_weights;
	int start_of_hidden_layers = start_of_weights + (num_per_input * num_per_layer);

	
	//input layer
	buffer[threadIdx.x] = 0;
	for (int i = 0; i < num_per_input; ++i)
	{
		buffer[threadIdx.x] += inputs_d[(blockIdx.x * num_per_input) + i] * Weights_D[start_of_weights + (threadIdx.x * num_per_input) + i];
	}
	buffer[threadIdx.x] = 1.0 / (1.0 + exp(-buffer[threadIdx.x] / response));
	__syncthreads();
	
	//subsequent hidden layers
	float temp;
	
	for (int i = 0; i < num_layers; ++i)
	{
		temp = 0;
		for (int j = 0; j < num_per_layer; ++j)
		{
			temp += buffer[j] * Weights_D[start_of_hidden_layers + (num_per_layer * num_per_layer * i) + (num_per_layer * threadIdx.x) + j];
		}
		temp = 1.0 / (1.0 + exp(-temp / response));

		__syncthreads();
		buffer[threadIdx.x] = temp;
		__syncthreads();
	}
	
	//output layer
	if (threadIdx.x < num_per_output)
	{
		temp = 0;
		for (int i = 0; i < num_per_layer; ++i)
		{
			temp += buffer[i] * Weights_D[start_of_hidden_layers + (num_per_layer * num_per_layer * num_layers) + (num_per_layer * threadIdx.x) + i];
		}
		temp = 1.0 / (1.0 + exp(-temp / response));

		__syncthreads();

		//copy the result back out to the outputs vector
		outputs_d[(blockIdx.x * num_per_output) + threadIdx.x] = temp;
	}
	
}