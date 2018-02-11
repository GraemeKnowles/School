#pragma once
#include "MiniGamePiece.h"
class MiniNinja :
	public MiniGamePiece
{
public:
	MiniNinja(GameBoard& gameboard, Player& player, GamePiece* demotedFrom = NULL) : MiniGamePiece(gameboard, player, "j", 1, demotedFrom) {}
	~MiniNinja() {}
	virtual std::vector<Move> getPossibleMoves() const;
};
