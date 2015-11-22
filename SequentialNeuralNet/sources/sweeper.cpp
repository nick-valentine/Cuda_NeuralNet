#include "../headers/sweeper.h"

const char *Sweeper::tex_name = "./img/tank.png";

Sweeper::Sweeper()
{
	mySprite.setTexture(*myMap.request(tex_name));
	mySprite.setScale(size / tex_size, size / tex_size);
	mySprite.setOrigin(size / 2, size / 2);
	
	
	pos_x = 0;
	pos_y = 0; 
	
	rotation = 0;
	
	score = 0;
}

Sweeper::Sweeper(int x, int y, parameters &my_params)
{
	mySprite.setTexture(*myMap.request(tex_name));
	mySprite.setScale(size / tex_size, size / tex_size);
	mySprite.setOrigin(tex_size / 2, tex_size / 2);
	
	this->my_params = my_params;
	
	//printf("\t\t\t\t\t\t\t%f %f\n", my_params.max_turn_rate, my_params.max_speed);
	
	
	pos_x = x;
	pos_y = y; 
	
	rotation = 0;
	
	score = 0;
}

std::vector<double> Sweeper::prepare_for_NN(int delta, std::vector<Mine> &mines)
{
	double closest_vec = 9999999;
	int closest_x = width;
	int closest_y = height;
	
	for(int i = 0; i < mines.size(); ++i)
	{
		int vec_x = mines[i].pos_x - pos_x;
		int vec_y = mines[i].pos_y - pos_y;
		double mine_vec = sqrt((vec_x * vec_x) + (vec_y * vec_y));
		if(mine_vec < closest_vec)
		{
			closest_x = mines[i].pos_x;
			closest_y = mines[i].pos_y;
			closest_vec = mine_vec;
		}
		
		if(abs(pos_x - mines[i].pos_x) < size &&
			abs(pos_y - mines[i].pos_y) < size)
		{
			mines[i].set_pos(rand() % width, rand() % height);
			//mines[i].set_pos(width / 2, height / 2);
			score++;
		}
	}
	
	//printf("%d %d\n", closest_x, closest_y);
	
	std::vector<double> inputs;
	inputs.push_back(closest_x);
	inputs.push_back(closest_y);
	inputs.push_back(pos_x);
	inputs.push_back(pos_y);
	
	return inputs;
}

void Sweeper::update(int delta, std::vector<double> controls)
{
	//printf("%d %d\n", pos_x, pos_y);
	//printf("%d %f %f\n", controls.size(), controls[0], controls[1]);
	
	//printf("\t\t\t\t\t\t\t%f %f\n", my_params.max_turn_rate, my_params.max_speed);
	
	//Rotate and drive controls
	if(my_params.drive_type == "car")
	{
		rotation += (2 * my_params.max_turn_rate * controls[0]) - my_params.max_turn_rate;
		pos_y += my_params.max_speed * controls[1] * sin((rotation * PI)/180);
		pos_x += my_params.max_speed * controls[1] * cos((rotation * PI)/180);
	}
	
	
	//x and y controls
	if(my_params.drive_type == "linear")
	{
		pos_y += (2 * my_params.max_speed * controls[0]) - my_params.max_speed;
		pos_x += (2 * my_params.max_speed * controls[1]) - my_params.max_speed;
	}
	
	
	//pos_y += controls[0] * my_params.max_speed;
	//pos_x += controls[1] * my_params.max_speed;
	
	//tank style controls
	if(my_params.drive_type == "tank")
	{
		double left_track = controls[0], right_track = controls[1];
		
		rotation += left_track - right_track;
		pos_y += my_params.max_speed * (1.0 - (left_track - right_track)) * sin((rotation * PI)/180);
		pos_x += my_params.max_speed * (1.0 - (left_track - right_track)) * cos((rotation * PI)/180);
	}
	
	//manage wraparound edges
	if(my_params.wraparound_edges == 1)
	{
		if(pos_y < -size)
		{
			pos_y = height + size;
		}
		if(pos_y > height + size)
		{
			pos_y = -size;
		}
		if(pos_x < -size)
		{
			pos_x = width + size;
		}
		if(pos_x > width + size)
		{
			pos_x = -size;
		}
	}
	
	//blocking edges
	if(my_params.wraparound_edges == 0)
	{
		if(pos_y < 0)
		{
			pos_y = 0;
		}
		if(pos_y > height)
		{
			pos_y = height;
		}
		if(pos_x < 0)
		{
			pos_x = 0;
		}
		if(pos_x > width)
		{
			pos_x = width;
		}
	}
}

void Sweeper::draw(sf::RenderWindow &window)
{
	mySprite.setRotation(rotation + 90);
	mySprite.setPosition(pos_x, pos_y);
	window.draw(mySprite);
}

void Sweeper::check_collision(Mine &test)
{
	if(abs(pos_x - test.pos_x) < size &&
		abs(pos_y - test.pos_y) < size)
		{
			test.set_pos(rand() % width, rand() % height);
			score++;
		}
}

int Sweeper::get_score()
{
	return score;
}

void Sweeper::reset_score()
{
	score = 0;
}

void Sweeper::set_position(int x, int y)
{
	pos_x = x;
	pos_y = y;
}