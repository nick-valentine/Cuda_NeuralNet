#include "../headers/TextureMap.h"

std::map<std::string, sf::Texture> TextureMap::tex_map;

TextureMap::TextureMap()
{

}

TextureMap::~TextureMap()
{

}
 	
sf::Texture* TextureMap::request(std::string file_name)
{
	if(this->tex_map.find(file_name) == tex_map.end())
	{
		//texture not found in map, load from file
		sf::Texture temp;
		if(!temp.loadFromFile(file_name))
		{
			//texture could not be loaded
			exit( 1 );
		}
		else
		{
			//texture loaded, add to map and return it
			tex_map[file_name] = temp;
			return &tex_map[file_name];
		}
	}
	else
	{
		//texture already loaded before
		return &tex_map[file_name];
	}
}

void TextureMap::clear()
{
	tex_map.clear();
}
