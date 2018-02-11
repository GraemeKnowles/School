#include "Ninja.h"
#include "GameBoard.h"
#include "Player.h"

Ninja::~Ninja()
{
}

std::vector<Move> Ninja::getPossibleMoves() const
{
	std::vector<Move> moves;

	const Position p = getPosition();
	const int xStart = p.getX(), yStart = p.getY();
	const int forwardDir = getOwner()->getAttackDirection();

	// Forward Right
	for (int x = xStart + 1, y = yStart + forwardDir;
		x < gameBoard->BOARD_WIDTH && y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(x, y) == NULL;
		++x, y += forwardDir)
	{
		GamePiece* piece = gameBoard->getPieceAt(x, y + forwardDir);
		if (piece != NULL && piece->getOwner() == getOwner())
		{
			piece = NULL;
		}
		moves.push_back(Move(p, Position(x, y), piece));
	}

	// Forward Left
	for (int x = xStart - 1, y = yStart + forwardDir;
		x >= 0 && y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(x, y) == NULL;
		--x, y += forwardDir)
	{
		GamePiece* piece = gameBoard->getPieceAt(x, y + forwardDir);
		if (piece != NULL && piece->getOwner() == getOwner())
		{
			piece = NULL;
		}

		moves.push_back(Move(p, Position(x, y), piece));
	}

	// Backward Right
	for (int x = xStart + 1, y = yStart - forwardDir;
		x < gameBoard->BOARD_WIDTH && y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(x, y) == NULL;
		++x, y -= forwardDir)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, y + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, y), piece));
		}
	}

	// Backward Left
	for (int x = xStart - 1, y = yStart - forwardDir;
		x >= 0 && y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(x, y) == NULL;
		--x, y -= forwardDir)
	{
		// Only valid if it's an attack
		GamePiece* piece = gameBoard->getPieceAt(x, y + forwardDir);
		if (piece != NULL && piece->getOwner() != getOwner())
		{
			moves.push_back(Move(p, Position(x, y), piece));
		}
	}

	return moves;
}