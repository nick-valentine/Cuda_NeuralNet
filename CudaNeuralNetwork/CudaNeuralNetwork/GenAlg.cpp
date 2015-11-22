#include "GenAlg.h"

struct GenomeComparer {
	bool operator() (Genome a, Genome b) { return (a < b); }
} myComparer;

Genome::Genome() : fitness(0)
{
}

Genome::Genome(std::vector<float> weights, float fitness) : weights(weights) , fitness(fitness)
{
}

bool Genome::operator<(const Genome &other)
{
	return(fitness < other.fitness);
}

GenAlg::GenAlg()
{
}

GenAlg::GenAlg(parameters &my_params , int number_weights) : my_params(my_params)
{
	pop_size = my_params.num_sweepers;
	mutation_rate = my_params.mutation_rate;
	crossover_rate = my_params.crossover_rate;
	weights_per_chromo = number_weights;
	total_fitness = 0;
	best_fitness = 0;
	average_fitness = 0;
	worst_fitness = 99999999;
	generation_count = 0;
	
	//generate random weights for initial population
	for(int i = 0; i < pop_size; ++i)
	{
		population.push_back(Genome());
		
		for(int j = 0; j < weights_per_chromo; ++j)
		{
			population[i].weights.push_back(small_rand());
		}
	}
}

std::vector<Genome> GenAlg::new_generation(std::vector<Genome> &last_generation)
{
	generation_count++;
	
	reset();
	
	std::sort(last_generation.begin(), last_generation.end(), myComparer);
	
	calculateStatistics();
	
	std::vector<Genome> new_population;
	
	//get elites
	GetBest(my_params.num_elite, my_params.num_copies_of_elite, last_generation, new_population);
	
	//genetic algorithm loop
	while(new_population.size() < pop_size)
	{
		Genome mother = getRandomChromosome();
		Genome father = getRandomChromosome();
		
		std::vector<float> child1, child2;
		
		crossover(mother.weights, father.weights, child1, child2);
		
		mutate(child1);
		mutate(child2);
		
		new_population.push_back(Genome(child1, 0));
		new_population.push_back(Genome(child2, 0));
	}
	
	population = new_population;
	
	return population;
}

std::vector<Genome> GenAlg::get_chromosomes()
{
	return population;
}

float GenAlg::get_avg_fitness()
{
	return average_fitness;
}

float GenAlg::get_best_fitness()
{
	return best_fitness;
}

float GenAlg::get_worst_fitness()
{
	return worst_fitness;
}

float GenAlg::get_total_fitness()
{
	return total_fitness;
}

int GenAlg::get_generation_count()
{
	return generation_count;
}

void GenAlg::crossover(const std::vector<float> &mother, const std::vector<float> &father,
						std::vector<float> &child1, std::vector<float> &child2)
{
	if(abs(small_rand()) > crossover_rate || mother == father)
	{
		child1 = mother;
		child2 = father;
	}
	else
	{
		int crossover_point = rand() % weights_per_chromo - 1;
		
		for(int i = 0; i < crossover_point; ++i)
		{
			child1.push_back(mother[i]);
			child2.push_back(father[i]);
		}
		for(int i = crossover_point; i < mother.size(); ++i)
		{
			child1.push_back(father[i]);
			child2.push_back(mother[i]);
		}
	}
}

void GenAlg::mutate(std::vector<float> &chromo)
{
	for(int i = 0; i < chromo.size(); ++i)
	{
		if(abs(small_rand()) < mutation_rate)
		{
			chromo[i] += small_rand() * my_params.perterbation;
		}
	}
}

Genome GenAlg::getRandomChromosome()
{
	//only return the best chromosomes
	return population[(rand() % (population.size()/8))];
}

void GenAlg::GetBest(int how_many, int number_of_copies, std::vector<Genome> &last_generation, std::vector<Genome> &best)
{
	for(int i = last_generation.size() - 1; i > last_generation.size() - how_many - 1; --i)
	{
		for(int j = 0; j < number_of_copies; ++j)
		{
			best.push_back(last_generation[i]);
		}
	}
}

void GenAlg::calculateStatistics()
{
	total_fitness = 0;
	float highest = 0;
	float lowest = 9999999;
	
	for(int i = 0; i < pop_size; ++i)
	{
		if(population[i].fitness > highest)
		{
			highest = population[i].fitness;
			best_genome = i;
			best_fitness = highest;
		}
		else if(population[i].fitness < lowest)
		{
			lowest = population[i].fitness;
			worst_fitness = lowest;
		}
		total_fitness += population[i].fitness;
	}
	
	average_fitness = total_fitness / pop_size;
}

void GenAlg::reset()
{
	total_fitness = 0;
	best_fitness = 0;
	worst_fitness = 9999999;
	average_fitness = 0;
}