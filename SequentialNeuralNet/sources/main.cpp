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

#include "../headers/Game.h"

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

int main()
{
	Game my_game;
	
	while(my_game.get_window()->isOpen())
	{
		my_game.handle_events();
		my_game.update();
		my_game.draw();
	}
	
	return 0;
}