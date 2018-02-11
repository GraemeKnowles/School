#include "Samurai.h"
#include "GameBoard.h"
#include "Player.h"

Samurai::~Samurai()
{
}

std::vector<Move> Samurai::getPossibleMoves() const
{
	std::vector<Move> moves;

	const Position p = getPosition();
	const int xStart = p.getX(), yStart = p.getY();
	const int forwardDir = getOwner()->getAttackDirection();

	// Right
	for (int x = xStart + 1; x < gameBoard->BOARD_WIDTH && gameBoard->getPieceAt(x, yStart) == NULL; ++x)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, yStart + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, yStart), piece));
		}
	}

	// Left
	for (int x = xStart - 1; x >= 0 && gameBoard->getPieceAt(x, yStart) == NULL; --x)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, yStart + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, yStart), piece));
		}
	}

	// Forward
	for (int y = yStart + forwardDir; y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(xStart, y) == NULL; y += forwardDir)
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