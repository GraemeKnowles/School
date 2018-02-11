#include "MiniNinja.h"
#include "GameBoard.h"
#include "Player.h"

std::vector<Move> MiniNinja::getPossibleMoves() const
{
	std::vector<Move> moves;

	const Position p = getPosition();
	const int xStart = p.getX(), yStart = p.getY();
	const int forwardDir = getOwner()->getAttackDirection();

	// Initialize for forward movement
	bool mustBeAttack = false;
	int x;
	int y = yStart + forwardDir;
	for (int i = 0; i < 4; ++i)
	{
		switch (i)
		{
		case 0:// forward left
			x = xStart - 1;
			break;
		case 1:// forward right
			x = xStart + 1;
			break;
		case 2://backward left
			   // Set for checking backward
			mustBeAttack = true;
			y = yStart - forwardDir;
			x = xStart - 1;
			break;
		case 3://backward right
			x = xStart + 1;
			break;
		}

		// Empty space within the board
		if (x >= 0 && x < gameBoard->BOARD_WIDTH && y >= 0 && y < gameBoard->BOARD_HEIGHT && gameBoard->getPieceAt(x, y) == NULL)
		{
			// Check if it's attacking
			GamePiece* piece = gameBoard->getPieceAt(x, y + forwardDir);
			if (piece != NULL)
			{
				// Prevent from attacking owner
				if (piece->getOwner() == getOwner())
				{
					if (mustBeAttack)
					{
						continue;
					}
					piece = NULL;
				}
			}
			else if (mustBeAttack)
			{
				continue;
			}

			moves.push_back(Move(p, Position(x, y), piece));
		}
	}

	return moves;
}