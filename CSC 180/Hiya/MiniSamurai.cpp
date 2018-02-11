#include "MiniSamurai.h"
#include "GameBoard.h"
#include "Player.h"

std::vector<Move> MiniSamurai::getPossibleMoves() const
{
	std::vector<Move> moves;

	const Position p = getPosition();
	const int xStart = p.getX(), yStart = p.getY();
	const int forwardDir = getOwner()->getAttackDirection();

	// Right
	int x = xStart + 1;
	if (x < gameBoard->BOARD_WIDTH && gameBoard->getPieceAt(x, yStart) == NULL)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, yStart + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, yStart), piece));
		}
	}

	// Left
	x = xStart - 1;
	if (x >= 0 && gameBoard->getPieceAt(x, yStart) == NULL)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, yStart + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, yStart), piece));
		}
	}

	// Forward
	int y = yStart + forwardDir;
	if (y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(xStart, y) == NULL)
	{
		GamePiece* piece = gameBoard->getPieceAt(xStart, y + forwardDir);
		if (piece != NULL && piece->getOwner() == getOwner())
		{
			piece = NULL;
		}
		moves.push_back(Move(p, Position(xStart, y), piece));
	}

	return moves;
}