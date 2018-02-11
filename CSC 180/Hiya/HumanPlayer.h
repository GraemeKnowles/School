#pragma once
#include "Player.h"
class HumanPlayer :
	public Player
{
public:
	HumanPlayer() {}
	~HumanPlayer() {}

	// Asks the player to input a move
	virtual Move getMove();
};
