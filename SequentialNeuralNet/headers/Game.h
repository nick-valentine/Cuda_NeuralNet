
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

#include "SNeuralNet.h"
#include "GenAlg.h"
#include "global.h"
#include "sweeper.h"
#include "mine.h"

class Game
{
public:
	Game();
	void handle_events();
	void update();
	void draw();
	
	sf::RenderWindow *get_window() const;
private:
	void load_params();

	const static unsigned int avg_length = 20;
	const static unsigned int frame_frequency = 33333; //1/30 of a second
	
	parameters my_params;
	static sf::RenderWindow window;
	sf::Clock nn_timer;
	sf::Clock tick_timer;
	sf::Clock generation_timer;
	
	GenAlg my_genetics;
	
	std::vector<NeuralNetwork> myBrains;
	std::vector<Sweeper> mySweepers;
	std::vector<Mine> mines;
	
	sf::Font myFont;
	sf::Text StatsText;
	
	double delta;
	double generation_time;
	
	bool frame_limit;
	bool last_Q;
	long int tick_counter;
	long int time_spent_on_NN;
	
	long int fps[avg_length];
	long int nn_time[avg_length];
	
	double avg_fps;
	double avg_time_on_NN;
};


#endif //GAME_H