
/*

Game class to manage the entire game.
Made into a class to make management of variables and game items overall a little easier.

*/

#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <new>

#include "GenAlg.h"
#include "global.h"
#include "TextureMap.h"
#include "NeuralNetKernel.h"
#include "PrepareForNN.h"
#include "UpdatePositions.h"

class Game
{
public:
	Game();
	~Game();
	void handle_events();
	void update();
	void draw();
	
	sf::RenderWindow *get_window() const;
private:
	void load_params();
	void next_generation();

	void prepare_sweeper_for_NN();
	void update_sweeper_positions();

	const static unsigned int avg_length = 20;
	const static unsigned int frame_frequency = 33333; //1/30 of a second
	const static unsigned int sweeper_size = 25;
	const static unsigned int texture_size = 100;
	const static char *tank_tex_name;
	const static char *mine_tex_name;
	
	parameters my_params;
	static sf::RenderWindow window;
	sf::Clock nn_timer;
	sf::Clock tick_timer;
	sf::Clock generation_timer;
	
	GenAlg my_genetics;
	
	//std::vector<NeuralNetwork> myBrains;
	//std::vector<Sweeper> mySweepers;
	//std::vector<Mine> mines;

	float * sweeper_rot_v;
	float * sweeper_pos_v;
	int * sweeper_score_v;
	float * mine_pos_v;

	TextureMap myMap;
	sf::Sprite tank_sprite;
	sf::Sprite mine_sprite;
	
	sf::Font myFont;
	sf::Text StatsText;
	
	float delta;
	float generation_time;
	
	bool frame_limit;
	bool last_Q;
	long int tick_counter;
	long int time_spent_on_NN;
	long int time_spent_getting_inputs;
	long int time_spend_updating_position;
	
	long int fps[avg_length];
	long int nn_time[avg_length];
	long int getting_inputs[avg_length];
	long int updating_pos[avg_length];
	
	float avg_fps;
	float avg_time_on_NN;
	float avg_getting_inputs;
	float avg_updating_pos;

	int weights_size;
	int weights_per_nn;
	float * weights;
	float * inputs;
	float * outputs;
};


#endif //GAME_H