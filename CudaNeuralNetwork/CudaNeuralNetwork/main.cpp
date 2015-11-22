/*

Nick Valentine

Main function for the sequential neural network.
This game demonstrates a sequential neural network controlling minesweepers gathering mines.
Every sweeper will have an identical neural network configured with config.ini
They each start with different randomly generated weights though. Their training is
done through a genetic algorithm which is also configurable. This algorithm will run
the game for a fixed amount of time before grabbing the highest scoring sweepers and mating them
generating a new pool of sweepers that are hopefully better than the last generation. This
process repeats indefinately.

Also press Q for fast forward.

*/

#include "Game.h"

/*
Parameters:
realActivationResponse: the activation response for the sigmoid function in the neural network
intHiddenLayers: the number of hidden layers in the neural network
intNeuronsPerHiddenLayer: the number of neurons per hidden layer in the neural network
realBias: the weight for the threshold in each neuron layer

realMaxTurnRate: the maximum turn rate for car mode
realMaxSpeed: the maximum speed for the cars

intNumMines: the number of mines to appear on the screen
intNumSweepers: the number of minesweepers to appear on the screen

realCrossoverWeight: the probability for two chromosomes to mix in the genetic algorithm
realMutationRate: the probability for a chromosome to be mutated
realPerterbation: the max amount that a weight in a chromosome is mutated
intNumTicksPerGeneration: the number of loops that occur before there is a new generation
intNumElite: take this number of top chromosomes to clone directly
intNumCopiesElite: for each elite to be cloned, clone this many times

stringDriveType: tank, car or linear
boolWraparoundEdges: 0 or 1 depending if the edges should allow wrapping around or block the sweeper

*/

#include "NeuralNetKernel.h"
#include "PrepareForNN.h"

int main()
{
	/*
	std::vector<float> output;

	float *inputs, *weights;
	inputs = (float *)malloc(45 * sizeof(float));
	weights = (float *)malloc(45 * 3 * sizeof(float));
	for (int i = 0; i < 45; ++i)
	{
		inputs[i] = ((float)(i) / 100.0);
	}
	for (int i = 0; i < 45 * 3; ++i)
	{
		weights[i] = ((float)(i) / 100.0);
	}

	output = call_cuda_neural_net(15, 3, 3, 3, 3, 45, 3, -1, 1, inputs, weights);

	for (int i = 0; i < output.size(); ++i)
	{
		printf("%f\n", output[i]);
	}

	char x;
	std::cin >> x;
	*/
	Game my_game;
	
	while(my_game.get_window()->isOpen())
	{
		my_game.handle_events();
		my_game.update();
		my_game.draw();
	}
	
	
	return 0;
}