#include "PrepareForNN.h"

float * g_sweepers_d, *g_mines_d, *g_distances_d, *g_inputs_d;
int * g_scores_d, *g_mineIdx_d;

void set_up_prepare_for_NN(int num_mines, int num_sweepers)
{
		cudaMalloc((void **)&g_sweepers_d, num_sweepers * 2 * sizeof(float));
		cudaMalloc((void **)&g_mines_d, num_mines * 2 * sizeof(float));
		cudaMalloc((void **)&g_distances_d, num_sweepers * num_mines * sizeof(float));
		cudaMalloc((void **)&g_inputs_d, num_sweepers * 4 * sizeof(float));
		cudaMalloc((void **)&g_scores_d, num_sweepers * sizeof(int));
		cudaMalloc((void **)&g_mineIdx_d, num_sweepers * num_mines * sizeof(int));
}

void end_prepare_for_NN()
{
		cudaFree(g_sweepers_d);
		cudaFree(g_mines_d);
		cudaFree(g_distances_d);
		cudaFree(g_inputs_d);
		cudaFree(g_scores_d);
		cudaFree(g_mineIdx_d);
}

void call_cuda_prepare_for_NN(float * sweeper_pos_v, float * mine_pos_v, float * inputs, int * sweeper_score_v, int num_sweepers, int num_mines, int width, int height, int size)
{

	cudaMemcpy(g_sweepers_d, sweeper_pos_v, num_sweepers * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_mines_d, mine_pos_v, num_mines * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_scores_d, sweeper_score_v, num_sweepers * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads(num_mines, 1, 1);
	dim3 blocks(1, num_sweepers, 1);
	calculate_distances <<<blocks, threads >>>(g_sweepers_d, g_mines_d, num_sweepers, num_mines, g_distances_d, g_inputs_d, g_scores_d, width, height, size);

#define BLOCK_SIZE 32
	int num_blocks_x = num_mines / (BLOCK_SIZE*2);
	if (num_mines % (BLOCK_SIZE * 2))
		num_blocks_x++;
	
	threads = dim3(BLOCK_SIZE, 1, 1);
	blocks = dim3(num_blocks_x, num_sweepers, 1);
	find_closest_mine <<<blocks, threads >>>(g_mines_d, g_distances_d, g_mineIdx_d, num_sweepers, num_mines, g_inputs_d);
	
	cudaMemcpy(inputs, g_inputs_d, num_sweepers * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(mine_pos_v, g_mines_d, num_mines * 2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(sweeper_score_v, g_scores_d, num_sweepers * sizeof(int), cudaMemcpyDeviceToHost);
#undef BLOCK_SIZE
}

__global__ void calculate_distances(float * sweeper_pos_v, float * mine_pos_v, int num_sweepers, int num_mines, float * distance_v, float * inputs, int * sweeper_score_v, int width, int height, int size)
{
#define sweeperIdx blockIdx.y
#define mineIdx threadIdx.x*2

	int distanceIdx = (blockIdx.y * num_mines) + threadIdx.x;
	float vec_x;
	float vec_y;
	float distance;

	__shared__ float sweeper_pos[2];

	if (threadIdx.x < 2)
	{
		sweeper_pos[threadIdx.x] = sweeper_pos_v[sweeperIdx + threadIdx.x];
		inputs[((sweeperIdx * 4) + threadIdx.x) + 2] = sweeper_pos[threadIdx.x]; //copy the sweeper position out to the inputs for the neural network in parallel

	}

	__syncthreads();


	vec_x = mine_pos_v[mineIdx] - sweeper_pos[0];
	vec_y = mine_pos_v[mineIdx + 1] - sweeper_pos[1];
	distance = sqrt((vec_x * vec_x) + (vec_y * vec_y));
	distance_v[distanceIdx] = distance;

	if (distance < size)
	{
		/*
		mine_pos_v[mineIdx] = width / 2;
		mine_pos_v[mineIdx + 1] = height / 2;
		*/
		
		mine_pos_v[mineIdx] = ((threadIdx.x + 1 ) * clock()) % width;
		mine_pos_v[mineIdx + 1] = ((threadIdx.x + 1) * clock()) % height;
		

		sweeper_score_v[sweeperIdx]++;
	}

#undef sweeperIdx
#undef mineIdx
}

__global__ void find_closest_mine(float * mine_pos_v, float * distances_v, int * mineIdx_v, int num_sweeprs, int num_mines, float * inputs)
{
#define sweeperIdx blockIdx.y
#define first_item blockIdx.y*num_mines
	int my_index = (gridDim.x * blockIdx.x) + threadIdx.x;

	//mineIdx_v[sweeperIdx * num_mines + threadIdx.x] = threadIdx.x;
	mineIdx_v[sweeperIdx * num_mines + my_index] = my_index;
	
	for (int stride = num_mines / 2; stride > 1; stride /= 2)
	{
		__syncthreads();
		if (my_index < stride)
		{
			if (distances_v[my_index + first_item] < distances_v[my_index + first_item + stride])
			{
				distances_v[my_index + first_item] = distances_v[my_index + first_item + stride];
				mineIdx_v[my_index + first_item] = mineIdx_v[my_index + first_item + stride];
			}
		}
	}

	inputs[sweeperIdx * 4] = mine_pos_v[mineIdx_v[sweeperIdx] * 2];
	inputs[sweeperIdx * 4 + 1] = mine_pos_v[mineIdx_v[sweeperIdx] * 2 + 1];

#undef sweeperIdx
#undef first_item
}