#include "../headers/SNeuralNet.h"

Neuron::Neuron(int numInputs) : numInputs(numInputs)
{
	//srand( time( NULL ) );
	//less than or equal to because of bias
	for(int i = 0; i <= numInputs; ++i)
	{
		weights.push_back(small_rand());
	}
}

NeuronLayer::NeuronLayer(int numNeurons, int numInputsPerNeuron) : numNeurons(numNeurons)
{
	for(int i = 0; i < numNeurons; ++i)
	{
		neurons.push_back(numInputsPerNeuron);
	}
}

NeuralNetwork::NeuralNetwork()
{
	
}

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, parameters &my_params) : numInputs(numInputs) , numOutputs(numOutputs) , my_params(my_params)
{
	numHiddenLayers  = my_params.num_hidden_layers;
	numNeuronsPerHiddenLayer = my_params.num_neurons_per_hidden_layer;
	create();
}

void NeuralNetwork::create()
{
	
	if(numHiddenLayers > 0)
	{
		//first layer will have a different number of inputs than the rest
		//because the number of input neurons might not match the number per
		//hidden layer
		layers.push_back(NeuronLayer(numNeuronsPerHiddenLayer, numInputs));
	
		//create rest of hidden layers
		for(int i = 1; i < numHiddenLayers; ++i)
		{
			layers.push_back(NeuronLayer(numNeuronsPerHiddenLayer, numNeuronsPerHiddenLayer));
		}
		
		//output layer
		layers.push_back(NeuronLayer(numOutputs, numNeuronsPerHiddenLayer));
	}
	else
	{
		//output layer if there are no hidden layers
		layers.push_back(NeuronLayer(numOutputs, numInputs));
	}

}

std::vector<double> NeuralNetwork::get_weights() const
{
	std::vector<double> weights;
	
	for(int i = 0; i < layers.size(); ++i)
	{
		for(int j = 0; j < layers[i].neurons.size(); ++j)
		{
			for(int k = 0; k < layers[i].neurons[j].weights.size(); ++k)
			{
				weights.push_back(layers[i].neurons[j].weights[k]);
			}
		}
	}
	
	return weights;
}

void NeuralNetwork::set_weights(std::vector<double> &weights)
{

	if(weights.size() != get_num_weights())
	{
		return;
	}
	int weight_counter = 0;
	
	for(int i = 0; i < layers.size(); ++i)
	{
		for(int j = 0; j < layers[i].neurons.size(); ++j)
		{
			for(int k = 0; k < layers[i].neurons[j].weights.size(); ++k)
			{
				layers[i].neurons[j].weights[k] = weights[weight_counter];
				weight_counter++;
			}
		}
	}
	return;
}

int NeuralNetwork::get_num_weights() const
{
	int weight_counter = 0;
	for(int i = 0; i < layers.size(); ++i)
	{
		for(int j = 0; j < layers[i].neurons.size(); ++j)
		{
			for(int k = 0; k < layers[i].neurons[j].weights.size(); ++k)
			{
				weight_counter++;
			}
		}
	}
	return weight_counter;
}

std::vector<double> NeuralNetwork::update(std::vector<double> &inputs)
{
	//buffer for the outputs from each layer and eventually the return value
	std::vector<double> outputs;
	
	int weight_counter = 0;
	
	if(inputs.size() != numInputs)
	{
		//the input is invalid, hopefully an empty vector is enough for error checking
		return outputs;
	}
	
	for(int i = 0; i < layers.size(); ++i)
	{
		if( i > 0 )
		{
			//result from the previous layer is now our inputs
			inputs = outputs;
		}
		
		outputs.clear();
		
		for(int j = 0; j < layers[i].neurons.size(); ++j)
		{
			weight_counter = 0;
			double sum_inputs = 0;
			
			int num_inputs = layers[i].neurons[j].weights.size();
			
			for(int k = 0; k < num_inputs; ++k)
			{
				sum_inputs += layers[i].neurons[j].weights[k] * inputs[weight_counter];
				//printf("WEIGHT: %f\t%f\n", layers[i].neurons[j].weights[k], inputs[weight_counter]);
				weight_counter++;
			}
			
			//add the bias
			sum_inputs += layers[i].neurons[j].weights[num_inputs - 1] * my_params.bias;
			
			outputs.push_back(sigmoid(sum_inputs, my_params.activation_response));
			
			//printf("\t\t\t%f %f %f\n", my_params.activation_response, sum_inputs, sigmoid(sum_inputs, my_params.activation_response));
		}
	}
	
	return outputs;
}