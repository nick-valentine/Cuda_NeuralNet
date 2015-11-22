#include "../headers/Game.h"

sf::RenderWindow Game::window;

Game::Game()
{
	load_params();
	
	printf("Parameters: \n");
	printf("Activation Response: %f\n", my_params.activation_response);
	printf("Number of Hidden Layers: %d\n", my_params.num_hidden_layers);
	printf("Number of Neurons Per Hidden Layer: %d\n", my_params.num_neurons_per_hidden_layer);
	printf("Bias: %f\n", my_params.bias);
	printf("Sweeper Max Turn Rate: %f\n", my_params.max_turn_rate);
	printf("Sweeper Max Speed: %f\n", my_params.max_speed);
	printf("Number of Mines: %d\n", my_params.num_mines);
	printf("Number of Sweepers: %d\n", my_params.num_sweepers);
	printf("Tank Controls: %s\n", my_params.drive_type.c_str());
	printf("Wraparound Edges: %d\n", my_params.wraparound_edges);
	printf("Crossover Rate: %f\n", my_params.crossover_rate);
	printf("Mutation Rate: %f\n", my_params.mutation_rate);
	printf("Perterbation: %f\n", my_params.perterbation);
	printf("Number of ticks per generation: %d\n", my_params.num_ticks);
	printf("Number of Elites: %d\n", my_params.num_elite);
	printf("Number of Copies of Elites: %d\n", my_params.num_copies_of_elite);
	
	window.create(sf::VideoMode(width, height), "Neural Net");
	
	for(int i = 0; i < my_params.num_sweepers; ++i)
	{
		myBrains.push_back(NeuralNetwork(4, 2, my_params));
	}
	for(int i = 0; i < my_params.num_sweepers; ++i)
	{
		mySweepers.push_back(Sweeper(rand() % width, rand() % height, my_params));
	}
	for(int i = 0; i < my_params.num_mines; ++i)
	{
		mines.push_back(Mine(rand() % width, rand() % height));
	}
	
	myFont.loadFromFile("Aaargh.ttf");
	StatsText = sf::Text("", myFont);
	StatsText.setCharacterSize(30);
	StatsText.setColor(sf::Color::White);
	
	frame_limit = true;
	last_Q = false;
	
	tick_counter = 0;
	time_spent_on_NN = 0;
	
	avg_fps = 0;
	avg_time_on_NN = 0;
	
	my_genetics = GenAlg(my_params, myBrains[0].get_num_weights());
	
	generation_timer.restart();
}

void Game::handle_events()
{
	//frame rate limiter
	if(sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
	{
		if(!last_Q)
		{
			frame_limit = !frame_limit;
		}
		last_Q = true;
	}
	else
	{
		last_Q = false;
	}

	//end frame rate limiter
	
	sf::Event event;
	while(window.pollEvent(event))
	{
		if(event.type == sf::Event::Closed)
		{
			window.close();
		}
	}
}

void Game::update()
{
	//limit framerate
	delta = tick_timer.restart().asMicroseconds();
	if(frame_limit)
	{
		sf::sleep(sf::microseconds(std::max(frame_frequency - delta, 0.0)));
	}

	//get inputs for neural networks from the sweepers
	std::vector<std::vector<double> > nn_vector;
	nn_vector.reserve(mySweepers.size());
	for(int i = 0; i < mySweepers.size(); ++i)
	{
		nn_vector.push_back(mySweepers[i].prepare_for_NN(delta, mines));
	}
	
	//get the outputs of the neural networks and time them
	time_spent_on_NN = 0;
	nn_timer.restart();
	for(int i = 0; i < myBrains.size(); ++i)
	{
		nn_vector[i] = myBrains[i].update(nn_vector[i]);
	}
	time_spent_on_NN = nn_timer.restart().asMicroseconds();
	
	//update the sweeper positions
	for(int i = 0; i < mySweepers.size(); ++i)
	{
		mySweepers[i].update(delta, nn_vector[i]);
	}
	
	//push and calculate averages
	nn_time[tick_counter % avg_length] = time_spent_on_NN;
	fps[tick_counter % avg_length] = delta;
	
	long int total_fps = 0, total_nn_time = 0;
	for(int i = 0; i < avg_length; ++i)
	{
		total_fps += fps[i];
		total_nn_time += nn_time[i];
	}
	avg_fps = total_fps / avg_length;
	avg_time_on_NN = total_nn_time / avg_length;
	//finished pushing and calculating averages
	
	tick_counter++;
	
	if(tick_counter >= my_params.num_ticks)
	{
		//new generation
		std::vector<Genome> my_genome, new_genome;
		for(int i = 0; i < mySweepers.size(); ++i)
		{
			my_genome.push_back(Genome(myBrains[i].get_weights(), mySweepers[i].get_score()));
			mySweepers[i].reset_score();
		}
		new_genome = my_genetics.new_generation(my_genome);
		
		for(int i = 0; i < mySweepers.size(); ++i)
		{
			myBrains[i].set_weights(new_genome[i].weights);
			mySweepers[i].set_position(rand() % width, rand() % height);
		}

		generation_time = generation_timer.restart().asSeconds();
		
		tick_counter = 0;
	}
}

void Game::draw()
{
	static std::stringstream ss;
	
	if(frame_limit || (!frame_limit && (tick_counter == 0)))
	{
		window.clear(sf::Color::Black);
	}
	if(frame_limit)
	{
		for(int i = 0; i < mySweepers.size(); ++i)
		{
			mySweepers[i].draw(window);
		}
		for(int i = 0; i < mines.size(); ++i)
		{
			mines[i].draw(window);
		}
	}
	if(frame_limit || (!frame_limit && (tick_counter == 0)))
	{
		ss<<"FPS: "<<1e6/(delta)<<"\n";
		ss<<"Microseconds on Neural Net: "<<time_spent_on_NN<<"\n";
		ss<<"Seconds on Last Generation: "<<generation_time<<"\n";
		ss<<"Generation: "<<my_genetics.get_generation_count()<<"\n";
		ss<<"Highest Fitness: "<<my_genetics.get_best_fitness()<<"\n";
		ss<<"Average Fitness: "<<my_genetics.get_avg_fitness()<<"\n";
		StatsText.setString(ss.str());
		window.draw(StatsText);
		ss.str(std::string());
		
		window.display();
	}
}

void Game::load_params()
{
	std::string setting, value;
	
	std::ifstream ifile;
	ifile.open("config.ini");
	
	ifile>>setting>>value;
	while(ifile.good())
	{
		if(setting == "realActivationResponse")
		{
			my_params.activation_response = atof(value.c_str());
		}
		else if(setting == "intHiddenLayers")
		{
			my_params.num_hidden_layers = atoi(value.c_str());
		}
		else if(setting == "intNeuronsPerHiddenLayer")
		{
			my_params.num_neurons_per_hidden_layer = atoi(value.c_str());
		}
		else if(setting == "realBias")
		{
			my_params.bias = atof(value.c_str());
		}
		else if(setting == "realMaxTurnRate")
		{
			my_params.max_turn_rate = atof(value.c_str());
		}
		else if(setting == "realMaxSpeed")
		{
			my_params.max_speed = atof(value.c_str());
		}
		else if(setting == "intNumMines")
		{
			my_params.num_mines = atoi(value.c_str());
		}
		else if(setting == "intNumSweepers")
		{
			my_params.num_sweepers = atoi(value.c_str());
		}
		else if(setting == "realCrossoverRate")
		{
			my_params.crossover_rate = atof(value.c_str());
		}
		else if(setting == "realMutationRate")
		{
			my_params.mutation_rate = atof(value.c_str());
		}
		else if(setting == "realPerterbation")
		{
			my_params.perterbation = atof(value.c_str());
		}
		else if(setting == "intNumTicksPerGeneration")
		{
			my_params.num_ticks = atoi(value.c_str());
		}
		else if(setting == "intNumElite")
		{
			my_params.num_elite = atoi(value.c_str());
		}
		else if(setting == "intNumCopiesElite")
		{
			my_params.num_copies_of_elite = atoi(value.c_str());
		}
		else if(setting == "stringDriveType")
		{
			my_params.drive_type = value; //linear, tank, car
		}
		else if(setting == "boolWraparoundEdges")
		{
			my_params.wraparound_edges = atoi(value.c_str());
		}
		
		ifile>>setting>>value;
	}
}

sf::RenderWindow *Game::get_window() const
{
	return &window;
}