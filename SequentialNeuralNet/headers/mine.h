
/*

Simple class to place mines on the screen that will be collected by the sweepers.
As of this time update is not used, the sweepers check the collisions themselves and
manage moving the mines when a collisions happens.

*/

#ifndef MINE_H
#define MINE_H

#include <SFML/Graphics.hpp>

#include <iostream>
#include <vector>
#include <math.h>
#include <string>

#include "global.h"
#include "TextureMap.h"

class Sweeper;

class Mine
{
friend class Sweeper;
public:
	Mine();
	Mine(int x, int y);
	
	void update(int delta);
	void draw(sf::RenderWindow &window);
	
	void set_pos(int x, int y);

private:
	const static double size = 25;
	const static double tex_size = 100;
	const static char *tex_name;

	TextureMap myMap;
	sf::Sprite mySprite;
	
	int pos_x;
	int pos_y;
};

#endif //MINE_H