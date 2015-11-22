/*

class to help load images from files and ensure that each file is loaded only once

when an image is loaded, it is saved in a static map of file name to the image
now when a texture is needed, create a map and use the request member function.


*/


#ifndef TEXTUREMAP_H
#define TEXTUREMAP_H

#include <SFML/Graphics.hpp>
#include <map>
#include <string>
#include <iostream>

class TextureMap
{
 public:
 	TextureMap();
 	~TextureMap();
 	
 	sf::Texture* request(std::string file_name);
 	static void clear();
 private:
 	static std::map<std::string, sf::Texture> tex_map;
};

#endif //TEXTUREMAP_H
