#include "../headers/mine.h"

const char *Mine::tex_name = "./img/mine.png";

Mine::Mine()
{
	mySprite.setTexture(*myMap.request(tex_name));
	mySprite.setScale(size / tex_size, size / tex_size);
	mySprite.setOrigin(size / 2, size / 2);
	
	pos_x = 0;
	pos_y = 0;
	
	mySprite.setPosition(pos_x, pos_y);
}

Mine::Mine(int x, int y)
{
	mySprite.setTexture(*myMap.request(tex_name));
	mySprite.setScale(size / tex_size, size / tex_size);
	mySprite.setOrigin(size / 2, size / 2);
	
	pos_x = x;
	pos_y = y;
	
	mySprite.setPosition(pos_x, pos_y);
}
	
void Mine::update(int delta)
{

}

void Mine::draw(sf::RenderWindow &window)
{
	window.draw(mySprite);
}

void Mine::set_pos(int x, int y)
{
	pos_x = x;
	pos_y = y;
	mySprite.setPosition(pos_x, pos_y);
}