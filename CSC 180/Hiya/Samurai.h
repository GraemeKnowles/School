#pragma once
#include "GamePiece.h"
#include "MiniSamurai.h"

class Samurai :
	public GamePiece
{
public:
	Samurai(GameBoard& gameboard, Player& player) : GamePiece(gameboard, player, "S", 4, new MiniSamurai(gameboard, player, this)) {}
	~Samurai();
	virtual std::vector<Move> getPossibleMoves() const;
};
