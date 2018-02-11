#include "HumanPlayer.h"
#include <iostream>
#include <string>
#include "Position.h"

// Asks the player to input a move
Move HumanPlayer::getMove()
{
	std::vector<Move> possibleMoves = getPossibleMoves();
	while (true)
	{
		// Get the Move
		std::cout << "Enter your move: ";
		char a, b, c, d;
		std::cin >> a >> b >> c >> d;
		std::string fromStr = std::string(1, a) + std::string(1, b), toStr = std::string(1, c) + std::string(1, d);

		Position from, to;
		bool invalid = false;
		if (Position::isValid(fromStr))
		{
			from = Position(fromStr);
		}
		else
		{
			std::cout << "First Position Invalid." << std::endl;
			invalid = true;
		}
		if (Position::isValid(toStr))
		{
			to = Position(toStr);
		}
		else
		{
			std::cout << "Second Position Invalid." << std::endl;
			invalid = true;
		}

		if (invalid) { continue; }

		Move enteredMove(from, to);

		//Validate the Move
		for (auto moveIt = possibleMoves.begin(); moveIt != possibleMoves.end(); ++moveIt)
		{
			// Check if move is valid
			if (enteredMove == *moveIt)
			{
				// If the move is an attack, print Hiya
				if (moveIt->getAttackedPiece() != NULL)
				{
					std::cout << "HIYA!" << std::endl;
				}

				// Return the Move
				return *moveIt;
			}
		}

		std::cout << "I'm sorry Dave, i'm afraid i can't do that." << std::endl;
	}
}