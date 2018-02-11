#pragma once
#include "GamePiece.h"
class King :
	public GamePiece
{
public:
	King(GameBoard& gameboard, Player& player) : GamePiece(gameboard, player, "K", 100, NULL) {}
	~King() {}
	virtual std::vector<Move> getPossibleMoves() const { return std::vector<Move>(); }
};
