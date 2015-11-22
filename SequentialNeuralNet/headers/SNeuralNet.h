
/*

The sequential neural network is composed of three classes

Neuron which is a list of weights. There is a weight for each of the neurons in the previous layer.
Neuron Layer which is a list of neurons.
Neural Network which does all of the work and contains a list of Neuron Layers.
Exactly how much of each Neuroin and Neuron Layer is defined in config.ini with a minimum of 6 per
Sweeper. Those 6 being the input and ouput layers which users have no control over.

In order to use the neural network, create a vector of doubles of length equal to the number of inputs
set when the network is created. Pass this vector into the update function and it will return
a vector of doubles of length equal to the number of outputs you set when constructing the network.

By default all of the weights for the neurons will be random, however get_weights and set_weights
can be used to get and set all of the weights in the network using 1 dimensional vectors of doubles.

*/

#ifndef SNEURALNET_H
#define SNEURALNET_H

#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>

#include "global.h"

inline double sigmoid(double activation, double response) {return (double)( 1.0 / (double)( 1.0 + exp(-activation / response)));}

class Neuron
{
public:
	Neuron(int numInputs);

	int numInputs;
	std::vector<double> weights;
};

class NeuronLayer
{
public:
	NeuronLayer(int numNeurons, int numInputsPerNeuron);
	
	int numNeurons;
	std::vector<Neuron> neurons;
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(int numInputs, int numOutputs, parameters &my_params);
	void create();
	
	int get_num_weights() const;
	std::vector<double> get_weights() const;
	
	void set_weights(std::vector<double> &weights);
	
	std::vector<double> update(std::vector<double> &inputs);
private:
	int numInputs;
	int numOutputs;
	int numHiddenLayers;
	int numNeuronsPerHiddenLayer;
	parameters my_params;
	std::vector<NeuronLayer> layers;
};

#endif