#include "Move.h"

unsigned long Move::weight = 0;

Move::Move(Position from, Position to, GamePiece* attacking, int score) : fromPos(from), toPos(to), attackedPiece(attacking), score(score)
{
	static const int SHIFT1 = 9;
	static const int SHIFT2 = 6;
	static const int SHIFT3 = 3;
	weightPtr = &weight;
	historyTableIndex = (from.getX() << SHIFT1) + (from.getY() << SHIFT2) + (from.getX() << SHIFT3) + from.getY();
}

bool Move::operator<(const Move & rhs) const
{
	// If they're both attacking moves, or both non-attacking moves
	if ((attackedPiece == NULL) == (rhs.attackedPiece == NULL))
	{
		return *weightPtr > *(rhs.weightPtr);// Sort by weight
	}

	return attackedPiece != NULL;// else, sort attacking move first
}

std::string Move::toString() const
{
	return fromPos.toString() + " to " + toPos.toString();
}