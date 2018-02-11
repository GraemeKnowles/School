#pragma once

#include <vector>
#include "GamePiece.h"
#include <unordered_map>

class Player
{
	friend class GameBoard;
public:
	Player() {}
	~Player();

	// Get the possible moves this player could make
	std::vector<Move> getPossibleMoves() const;
	// Get the move this player wants to make
	virtual Move getMove() = 0;

	// Direction this player attacks
	int getAttackDirection() { return attackingDirection; }

	// Pieces this player owns
	std::unordered_map<int, GamePiece*>* getOwnedPieces() const { return ownedPieces; }

	inline void setOpponent(Player& opp) { opponent = &opp; }
	inline Player* getOpponent() const { return opponent; }

	inline void setBoard(GameBoard& board) { this->board = &board; }
	inline GameBoard* getBoard() const { return board; }

private:
	std::unordered_map<int, GamePiece*>* ownedPieces;
	int attackingDirection;
	GameBoard* board;
	Player* opponent;
};
