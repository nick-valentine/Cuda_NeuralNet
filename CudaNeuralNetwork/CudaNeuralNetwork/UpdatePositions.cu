#include "UpdatePositions.h"

float * g_outputs_d, *g_sweepers_d_2;

void set_up_update_positions(int num_sweepers)
{
	cudaMalloc((void **)&g_outputs_d, num_sweepers * 2 * sizeof(float));
	cudaMalloc((void **)&g_sweepers_d_2, num_sweepers * 2 * sizeof(float));
}

void end_update_positions()
{
	cudaFree(g_outputs_d);
	cudaFree(g_sweepers_d_2);
}

void call_cuda_update_positions(int num_sweepers, float max_speed, float * outputs, float * sweepers)
{
	cudaMemcpy(g_outputs_d, outputs, num_sweepers * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sweepers_d_2, sweepers, num_sweepers * 2 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(2, 1, 1);
	dim3 blocks(num_sweepers, 1, 1);
	update_positions <<<blocks, threads >>>(max_speed, g_outputs_d, g_sweepers_d_2);

	cudaMemcpy(sweepers, g_sweepers_d_2, num_sweepers * 2 * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void update_positions(float max_speed, float * outputs_d, float * sweepers_d)
{
	int my_index = blockIdx.x * blockDim.x + threadIdx.x;

	sweepers_d[my_index] +=  (2 * outputs_d[my_index] * max_speed) - max_speed;
}