#pragma once
#include "MiniGamePiece.h"
class MiniSamurai :
	public MiniGamePiece
{
public:
	MiniSamurai(GameBoard& gameboard, Player& player, GamePiece* demotedFrom = NULL) : MiniGamePiece(gameboard, player, "s", 1, demotedFrom) {}
	~MiniSamurai() {}
	virtual std::vector<Move> getPossibleMoves() const;
};
