#include "GamePiece.h"

GamePiece::GamePiece(GameBoard & gameBoard, Player & owner, std::string boardRep, int value, GamePiece * demoteTo) 
	: gameBoard(&gameBoard), owner(&owner), boardRepresentation(boardRep), value(value), demotePieceTo(demoteTo)
{
	// Sets up a unique ID for every game piece created
	static int id = 0;
	uniqueID = id++;
}

GamePiece::~GamePiece()
{
	if (demotePieceTo != NULL)
	{
		delete demotePieceTo;
	}
}