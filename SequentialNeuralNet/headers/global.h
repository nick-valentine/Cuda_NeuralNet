
/*

Just a few usefull items to the game.
Height and Width are the dimensions of the window that is created
PI is pi for the Sweepers to because not every function is in degrees

*/

#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdlib.h>
#include <math.h>
#include <string>

const static unsigned int height = 720;
const static unsigned int width = 1280;

#define PI 3.14159265

//returns a random float between -1 and 1
inline double small_rand() {return (2 * (((double)(rand() % 1000))/1000)) - 1;}

struct parameters
{
	//neural network
	double activation_response;
	int num_hidden_layers;
	int num_neurons_per_hidden_layer;
	double bias;
	
	//sweepers
	double max_turn_rate;
	double max_speed;
	
	//game
	int num_mines;
	int num_sweepers;
	
	//genetics
	double crossover_rate;
	double mutation_rate;
	double perterbation;
	int num_ticks;
	
	int num_elite;
	int num_copies_of_elite;
	
	std::string drive_type;
	int wraparound_edges;
};



#endif //GLOBAL_H