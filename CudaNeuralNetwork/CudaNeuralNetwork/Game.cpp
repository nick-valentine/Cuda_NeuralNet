#include "Game.h"

const char *Game::tank_tex_name = "./img/tank.png";
const char *Game::mine_tex_name = "./img/mine.png";

sf::RenderWindow Game::window;

#define USE_CUDA false

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

	//number * 2 (x and y)
	sweeper_rot_v = (float *)malloc(my_params.num_sweepers * sizeof(float));
	sweeper_pos_v = (float *)malloc(my_params.num_sweepers * 2 * sizeof(float));
	sweeper_score_v = (int *)malloc(my_params.num_sweepers * sizeof(int));
	mine_pos_v = (float *)malloc(my_params.num_mines * 2 * sizeof(float));

	for (int i = 0; i < my_params.num_mines; ++i)
	{
		mine_pos_v[i * 2] = rand() % width;
		mine_pos_v[i * 2 + 1] = rand() % height;
	}

	for (int i = 0; i < my_params.num_sweepers; ++i)
	{
		sweeper_rot_v[i] = rand() % 360;
		sweeper_pos_v[i * 2] = rand() % width;
		sweeper_pos_v[i * 2 + 1] = rand() % height;
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
	
	//see documentation at top of kernel.cu
	weights_per_nn = (4 * my_params.num_neurons_per_hidden_layer) + (my_params.num_neurons_per_hidden_layer * (my_params.num_hidden_layers - 1)) + (2 * my_params.num_neurons_per_hidden_layer);
	weights_size = weights_per_nn * my_params.num_sweepers;
	my_genetics = GenAlg(my_params, weights_per_nn);
	
	weights = (float *)malloc(weights_size * sizeof(float));
	inputs = (float *)malloc(my_params.num_sweepers * 4 * sizeof(float));
	outputs = (float *)malloc(my_params.num_sweepers * 2 * sizeof(float));
	
	for (int i = 0; i < weights_size; ++i)
	{
		weights[i] = small_rand();
	}

	tank_sprite.setTexture(*myMap.request(tank_tex_name));
	tank_sprite.setScale((double)sweeper_size / (double)texture_size, (double)sweeper_size / (double)texture_size);
	tank_sprite.setOrigin((double)texture_size / 2.0, (double)texture_size / 2.0);

	mine_sprite.setTexture(*myMap.request(mine_tex_name));
	mine_sprite.setScale((double)sweeper_size / (double)texture_size, (double)sweeper_size / (double)texture_size);
	mine_sprite.setOrigin((double)texture_size / 2.0, (double)texture_size / 2.0);

	start_cuda(weights_size, weights);
	set_up_prepare_for_NN(my_params.num_mines, my_params.num_sweepers);
	set_up_update_positions(my_params.num_sweepers);

	generation_timer.restart();
}

Game::~Game()
{
	free(sweeper_rot_v);
	free(sweeper_pos_v);
	free(sweeper_score_v);
	free(mine_pos_v);
	free(weights);
	free(inputs);
	free(outputs);
	end_cuda();
	end_prepare_for_NN();
	end_update_positions();
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
	if (frame_limit)
	{
		sf::sleep(sf::microseconds((frame_frequency - delta > 0.0) ? frame_frequency - delta : 0.0));
	}

	//get inputs for neural networks from the sweepers
	nn_timer.restart();
	if (USE_CUDA)
	{
		call_cuda_prepare_for_NN(sweeper_pos_v, mine_pos_v, inputs, sweeper_score_v, my_params.num_sweepers, my_params.num_mines, width, height, sweeper_size + sweeper_size);
	}
	else
	{
		prepare_sweeper_for_NN();
	}
	time_spent_getting_inputs = nn_timer.restart().asMicroseconds();

	//4 inputs and 2 outputs then all the hidden layers
	int num_per_sweeper = 4 + 2 + (my_params.num_hidden_layers * my_params.num_neurons_per_hidden_layer);
	
	//get the outputs of the neural networks and time them
	time_spent_on_NN = 0;
	nn_timer.restart();
	
	call_cuda_neural_net(num_per_sweeper, my_params.num_neurons_per_hidden_layer, 4, 2, my_params.num_sweepers, weights_per_nn, my_params.num_hidden_layers, my_params.activation_response, inputs, outputs);

	time_spent_on_NN = nn_timer.restart().asMicroseconds();
	
	//update the sweeper positions
	nn_timer.restart();
	if (USE_CUDA)
	{
		call_cuda_update_positions(my_params.num_sweepers, my_params.max_speed, outputs, sweeper_pos_v);
	}
	else
	{
		update_sweeper_positions();
	}
	time_spend_updating_position = nn_timer.restart().asMicroseconds();

	//push and calculate averages
	nn_time[tick_counter % avg_length] = time_spent_on_NN;
	fps[tick_counter % avg_length] = delta;
	getting_inputs[tick_counter % avg_length] = time_spent_getting_inputs;
	updating_pos[tick_counter % avg_length] = time_spend_updating_position;
	
	long int total_fps = 0, total_nn_time = 0, total_inputs_time = 0, total_update_pos_time = 0;
	for(int i = 0; i < avg_length; ++i)
	{
		total_fps += fps[i];
		total_nn_time += nn_time[i];
		total_inputs_time += getting_inputs[i];
		total_update_pos_time += updating_pos[i];

	}
	avg_fps = total_fps / avg_length;
	avg_time_on_NN = total_nn_time / avg_length;
	avg_getting_inputs = total_inputs_time / avg_length;
	avg_updating_pos = total_update_pos_time / avg_length;
	//finished pushing and calculating averages
	
	tick_counter++;
	
	if(tick_counter >= my_params.num_ticks)
	{
		next_generation();
		//print_weights(weights_size);
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
		for (int i = 0; i < my_params.num_sweepers; ++i)
		{
			tank_sprite.setPosition(sweeper_pos_v[i * 2], sweeper_pos_v[i * 2 + 1]);
			tank_sprite.setRotation(sweeper_rot_v[i]);
			window.draw(tank_sprite);
		}

		for (int i = 0; i < my_params.num_mines; ++i)
		{
			mine_sprite.setPosition(mine_pos_v[i * 2], mine_pos_v[i * 2 + 1]);
			window.draw(mine_sprite);
		}

	}
	if(frame_limit || (!frame_limit && (tick_counter == 0)))
	{
		ss<<"FPS: "<<1e6/(delta)<<"\n";
		ss<<"Microseconds on Neural Net: "<<time_spent_on_NN<<"\n";
		ss << "Microseconds on Inputs: " << avg_getting_inputs << "\n";
		ss << "Microseconds on Updating Pos: " << avg_updating_pos << "\n";
		ss<<"Seconds on Last Generation: "<<generation_time<<"\n";
		ss<<"Generation: "<<my_genetics.get_generation_count()<<"\n";
		ss<<"Highest Fitness: "<<my_genetics.get_best_fitness()<<"\n";
		ss<<"Average Fitness: "<<my_genetics.get_avg_fitness()<<"\n";
		ss << "Ticks to epoch: " << my_params.num_ticks - tick_counter << "\n";
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

void Game::next_generation()
{
	//new generation
	std::vector<Genome> my_genome, new_genome;
	std::vector<float> weights_v;
	int num_weights_per_sweeper = weights_per_nn;
	weights_v.resize(num_weights_per_sweeper);

	for (int i = 0; i < my_params.num_sweepers; ++i)
	{
		for (int j = 0; j < num_weights_per_sweeper; ++j)
		{
			weights_v[j] = weights[i * num_weights_per_sweeper + j];
		}
		my_genome.push_back(Genome(weights_v, sweeper_score_v[i]));
		sweeper_score_v[i] = 0;
	}

	new_genome = my_genetics.new_generation(my_genome);

	for (int i = 0; i < my_params.num_sweepers; ++i)
	{
		//myBrains[i].set_weights(new_genome[i].weights);
		for (int j = 0; j < new_genome[i].weights.size(); ++j)
		{
			weights[i * new_genome[0].weights.size() + j] = new_genome[i].weights[j];
			//weights[i * new_genome[0].weights.size() + j] = small_rand();
		}

		sweeper_pos_v[i * 2] = rand() % width;
		sweeper_pos_v[i * 2 + 1] = rand() % height;
	}

	copy_weights(weights_size, weights);

	generation_time = generation_timer.restart().asSeconds();

	tick_counter = 0;
}

void Game::prepare_sweeper_for_NN()
{
	for (int i = 0; i < my_params.num_sweepers; i++)
	{
		float closest_vec = 999999999;
		int closest_x = width;
		int closest_y = height;

		for (int j = 0; j < my_params.num_mines; j++)
		{
			int vec_x = mine_pos_v[j * 2] - sweeper_pos_v[i * 2];
			int vec_y = mine_pos_v[j * 2 + 1] - sweeper_pos_v[i * 2 + 1];

			float mine_vec = sqrt((vec_x * vec_x) + (vec_y * vec_y));
			if (mine_vec < closest_vec)
			{
				closest_x = mine_pos_v[j * 2];
				closest_y = mine_pos_v[j * 2 + 1];
				closest_vec = mine_vec;
			}

			if (abs(sweeper_pos_v[i * 2] - mine_pos_v[j * 2]) < sweeper_size &&
				abs(sweeper_pos_v[i * 2 + 1] - mine_pos_v[j * 2 + 1]) < sweeper_size)
			{
				mine_pos_v[j * 2] = rand() % width;
				mine_pos_v[j * 2 + 1] = rand() % height;

				sweeper_score_v[i]++;
			}
		}

		inputs[i * 4    ] = closest_x;
		inputs[i * 4 + 1] = closest_y;
		inputs[i * 4 + 2] = sweeper_pos_v[i * 2];
		inputs[i * 4 + 3] = sweeper_pos_v[i * 2 + 1];
	}
}

void Game::update_sweeper_positions()
{
	/*
	for (int i = 0; i < my_params.num_sweepers; ++i)
	{
		printf("%f\t%f\n", outputs[2 * i], outputs[2 * i + 1]);
	}
	*/

	for (int i = 0; i < my_params.num_sweepers; i++)
	{
		//Rotate and drive controls
		if (my_params.drive_type == "car")
		{
			sweeper_rot_v[i] += (2 * my_params.max_turn_rate * outputs[i * 2]) - my_params.max_turn_rate;
			sweeper_pos_v[i * 2 + 1] += my_params.max_speed * outputs[i * 2 + 1] * sin((sweeper_rot_v[i] * PI) / 180);
			sweeper_pos_v[i * 2] += my_params.max_speed * outputs[i * 2 + 1] * cos((sweeper_rot_v[i] * PI) / 180);
		}


		//x and y controls
		if (my_params.drive_type == "linear")
		{
			sweeper_pos_v[i * 2 + 1] += (2 * my_params.max_speed * outputs[i * 2]) - my_params.max_speed;
			sweeper_pos_v[i * 2] += (2 * my_params.max_speed * outputs[i * 2 + 1]) - my_params.max_speed;
		}


		//sweeper_pos_v[i*2+1] += controls[0] * my_params.max_speed;
		//sweeper_pos_v[i*2] += controls[1] * my_params.max_speed;

		//tank style controls
		if (my_params.drive_type == "tank")
		{
			float left_track = outputs[i * 2], right_track = outputs[i * 2 + 1];

			sweeper_rot_v[i] += left_track - right_track;
			sweeper_pos_v[i * 2 + 1] += my_params.max_speed * (1.0 - (left_track - right_track)) * sin((sweeper_rot_v[i] * PI) / 180);
			sweeper_pos_v[i * 2] += my_params.max_speed * (1.0 - (left_track - right_track)) * cos((sweeper_rot_v[i] * PI) / 180);
		}

		//manage wraparound edges
		if (my_params.wraparound_edges == 1)
		{
			if (sweeper_pos_v[i * 2 + 1] < -sweeper_size)
			{
				sweeper_pos_v[i * 2 + 1] = height + sweeper_size;
			}
			if (sweeper_pos_v[i * 2 + 1] > height + sweeper_size)
			{
				sweeper_pos_v[i * 2 + 1] = -sweeper_size;
			}
			if (sweeper_pos_v[i * 2] < -sweeper_size)
			{
				sweeper_pos_v[i * 2] = width + sweeper_size;
			}
			if (sweeper_pos_v[i * 2] > width + sweeper_size)
			{
				sweeper_pos_v[i * 2] = -sweeper_size;
			}
		}

		//blocking edges
		if (my_params.wraparound_edges == 0)
		{
			if (sweeper_pos_v[i * 2 + 1] < 0)
			{
				sweeper_pos_v[i * 2 + 1] = 0;
			}
			if (sweeper_pos_v[i * 2 + 1] > height)
			{
				sweeper_pos_v[i * 2 + 1] = height;
			}
			if (sweeper_pos_v[i * 2] < 0)
			{
				sweeper_pos_v[i * 2] = 0;
			}
			if (sweeper_pos_v[i * 2] > width)
			{
				sweeper_pos_v[i * 2] = width;
			}
		}
	}
}

sf::RenderWindow *Game::get_window() const
{
	return &window;
}