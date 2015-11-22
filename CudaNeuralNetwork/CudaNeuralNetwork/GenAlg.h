
/*

The genetic algorithm uses a class Genome which maps genomes (1d vectors of neural network weights)
to fitnesses.

To use the genetic algorithm, once it is constructed, create a vector of Genomes which should be the 
Genomes for the entire current population and pass them into the new_generation member function
This function takes these genomes and sorts them based on fitnesses. Then depending on 
intNumElite and intNumCopiesElite in config.ini, it will take the NumElite top Genomes
and make NumCopiesElite of each of these genomes, then push them into the vector of Genomes
that will be the next generation. After this until there are intNumSweepers Genomes, two
Genomes from the top third of the population are chosen at random and bred then pushed into the vector.
This vector is returned and you may insert these new Genes into your set of neural networks.
Hopefully it will be better than the last generation.

Breeding is composed of two parts:
Two children Genomes(vectors of floats) are made.
a random number is created with a max of realCrossoverRate. This is
the how far the first loop goes in the first loop. The first loop takes the contents of the mother
Genome and places it into child 1 as well as taking the contents of the father and placing it into 
child 2.
So after the first loop child 1 will have the first random number % of the mother and the same for
child 2 and the father.
There is then a second loop that places the rest of the mother's genome into child 2 and the rest
of the father's genome into child 1.

Now there are two children and they may have mutations:
Each child get a chance to be mutated where their Genome (a vector of floats) is passed into 
a loop where for each item in this vector, a random number is generated, if it is larger than
realMutationRate, then this item will be mutated. It has a random number between -realPerterbation
and realPerterbation added onto it as the mutation.

Breeding is now done and there are 2 new (hopefully) functional children

*/

#ifndef GENALG_H
#define GENALG_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "global.h"

class Genome
{
public:
	Genome();
	Genome(std::vector<float> weights, float fitness);
	
	bool operator<(const Genome &other);
	
	std::vector<float> weights;
	float fitness;
};

class GenAlg
{
public:
	GenAlg();
	GenAlg(parameters &my_params, int number_weights);
	std::vector<Genome> new_generation(std::vector<Genome> &last_generation);
	
	std::vector<Genome> get_chromosomes();
	float get_avg_fitness();
	float get_best_fitness();
	float get_worst_fitness();
	float get_total_fitness();
	int get_generation_count();
private:
	std::vector<Genome> population;
	int pop_size;
	int weights_per_chromo;
	float total_fitness;
	float best_fitness;
	float average_fitness;
	float worst_fitness;
	int best_genome;
	float mutation_rate;
	float crossover_rate;
	int generation_count;
	
	parameters my_params;
	
	void crossover(const std::vector<float> &mother, const std::vector<float> &father, 
					std::vector<float> &child1, std::vector<float> &child2);
					
	void mutate(std::vector<float> &chromo);
	
	Genome getRandomChromosome();
	
	void GetBest(int how_many, int number_of_copies, std::vector<Genome> &last_generation, std::vector<Genome> &best);
	
	void calculateStatistics();
	void reset();
};


#endif //GENALG_H