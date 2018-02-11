#pragma once
#include "GamePiece.h"
class MiniGamePiece :
	public GamePiece
{
public:
	MiniGamePiece(GameBoard& gameboard, Player& player, std::string displayString, int value, GamePiece* demotedFrom) 
		: GamePiece(gameboard, player, displayString, value, NULL), demotedFrom(demotedFrom) {}
	~MiniGamePiece() {}

private:
	GamePiece* demotedFrom;
};
