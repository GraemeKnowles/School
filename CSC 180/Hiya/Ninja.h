#pragma once
#include "GamePiece.h"
#include "MiniNinja.h"

class Ninja :
	public GamePiece
{
public:
	Ninja(GameBoard& gameboard, Player& player) : GamePiece(gameboard, player, "J", 6, new MiniNinja(gameboard, player, this)) {}
	~Ninja();
	virtual std::vector<Move> getPossibleMoves() const;
};
