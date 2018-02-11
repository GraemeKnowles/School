#include "GameBoard.h"
#include "AIPlayer.h"
#include "HumanPlayer.h"

#include <iostream>
#include <windows.h>

int getTurnOrder();
void setPriority();

int main()
{
	// Sets the priority in the operating system
	setPriority();

	// The players
	//AIPlayer human(5000);
	HumanPlayer human;
	AIPlayer ai(5000);
	// The Board the game is played on
	GameBoard board(human, ai);
	ai.createBoardCopies();
	//human.createBoardCopies();

	// Print board
	board.print();

	// Who goes first?
	int playerIndex = getTurnOrder();

	// So it's possibile to cycle through the player
	std::vector<Player*> players;
	players.push_back(&human);
	players.push_back(&ai);
	// Main game loop
	for (unsigned int i = playerIndex; !board.hasEnded(); ++i)
	{
		// Get the player whose turn it is
		Player* player = players[i % players.size()];

		// Check for no more moves loss
		if (player->getPossibleMoves().size() == 0)
		{
			board.setLoser(player);
			break;
		}

		// Get Move
		Move move = player->getMove();

		// Make Move
		board.confirmMove(move);
		ai.updateBoardCopies(move);
		//human.updateBoardCopies(move);

		// Display Board
		board.print();
	}

	// if Game Over then announce
	if (board.getWinner() == &ai)
	{
		std::cout << "The computer has won in " << board.getConfirmedMovesCount() << ", better luck next time." << std::endl;
	}
	else
	{
		std::cout << "You have won in " << board.getConfirmedMovesCount() << ", Congratulations!" << std::endl;
	}
}

// Gets which player goes first
int getTurnOrder()
{
	std::cout << "Who should go first? c for computer, anything else for human: ";
	char first;
	std::cin >> first;
	std::cin.clear();
	first = tolower(first);
	return first == 'c' ? 1 : 0;
}

void setPriority()
{
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
}