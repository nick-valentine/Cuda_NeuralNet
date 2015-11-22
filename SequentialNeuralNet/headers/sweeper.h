
/*

Minesweeper class to be controlled by neural networks.
The neural network has been moved out of this class so that they may all be processed 
at the same time making timing and moving to CUDA easier. The neural network they use 
has 4 inputs: the closest mine's x and y position, and their own x and y position. Then
there are 2 outputs which change function based on what is set in config.ini.
If stringDriveType is set to linear, the outputs will be the amount to move in the x and y direction.
If it is car, the outputs will be speed and amount to steer respectively.
If it is tank, the outputs will be power to left track and power to right track.

Score for the genetic algorithm is gained from picking up a mine, when a mine is picked up
the sweeper will place the mine at a random location on the screen.

*/

#ifndef SWEEPER_H
#define SWEEPER_H

#include <SFML/Graphics.hpp>

#include <iostream>
#include <vector>
#include <math.h>
#include <string>

#include "TextureMap.h"
#include "global.h"
#include "mine.h"

class Sweeper
{
public:
	Sweeper();
	Sweeper(int x, int y, parameters &my_params);
	
	std::vector<double> prepare_for_NN(int delta, std::vector<Mine> &mines);
	void update(int delta, std::vector<double> controls);
	void draw(sf::RenderWindow &window);
	
	void check_collision(Mine &test);
	
	int get_score();
	void reset_score();
	
	void set_position(int x, int y);
private:
	const static unsigned int speed = 5;
	const static unsigned int rot_speed = 5;
	const static double size = 25;
	const static double tex_size = 100;
	const static char *tex_name;

	TextureMap myMap;
	sf::Sprite mySprite;
	
	parameters my_params;
	
	int pos_x;
	int pos_y;
	
	double rotation;
	
	int score;
};

#endif //SWEEPER_H