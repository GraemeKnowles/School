#include "Player.h"
#include <iterator>

Player::~Player()
{
}

std::vector<Move> Player::getPossibleMoves() const
{
	std::vector<Move> moves;
	// iterates through all of this player's pieces and gets their possible moves
	for (auto piecesIt = ownedPieces->begin(); piecesIt != ownedPieces->end(); ++piecesIt)
	{
		auto possibleMoves = piecesIt->second->getPossibleMoves();
		for (auto movesIt = possibleMoves.begin(); movesIt != possibleMoves.end(); ++movesIt)
		{
			moves.push_back(*movesIt);
		}
	}

	return moves;
}