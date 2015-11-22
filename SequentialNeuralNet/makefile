
all: build/main.o build/TextureMap.o build/sweeper.o build/mine.o build/SNeuralNet.o build/GenAlg.o build/Game.o
	g++ -o main build/main.o build/TextureMap.o build/sweeper.o build/mine.o build/SNeuralNet.o build/GenAlg.o build/Game.o -lsfml-graphics -lsfml-window -lsfml-system

windows: build/main.o build/TextureMap.o build/sweeper.o build/mine.o build/SNeuralNet.o build/GenAlg.o build/Game.o
	g++ -o main.exe build/main.o build/TextureMap.o build/sweeper.o build/mine.o build/SNeuralNet.o build/GenAlg.o  build/Game.o -lsfml-graphics -lsfml-window -lsfml-system

winclean:
	rm build/*
	
clean:
	rm ./build/*.o
	
build/main.o: sources/main.cpp
	g++ -c sources/main.cpp -o build/main.o
	
build/TextureMap.o: sources/TextureMap.cpp headers/TextureMap.h
	g++ -c sources/TextureMap.cpp -o build/TextureMap.o
	
build/sweeper.o: sources/sweeper.cpp headers/sweeper.h
	g++ -c sources/sweeper.cpp -o build/sweeper.o
	
build/mine.o: sources/mine.cpp headers/mine.h
	g++ -c sources/mine.cpp -o build/mine.o
	
build/SNeuralNet.o: sources/SNeuralNet.cpp headers/SNeuralNet.h
	g++ -c sources/SNeuralNet.cpp -o build/SNeuralNet.o
	
build/GenAlg.o: sources/GenAlg.cpp headers/GenAlg.h
	g++ -c sources/GenAlg.cpp -o build/GenAlg.o
	
build/Game.o: sources/Game.cpp headers/Game.h
	g++ -c sources/Game.cpp -o build/Game.o