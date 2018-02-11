#pragma once

#include "GamePiece.h"
#include <vector>
#include <stack>
#include <unordered_map>


// Class that controls the state of the board. Provides functionality
// for moving pieces and checking the game state (won/loss)
class GameBoard
{
public:
	static const int BOARD_HEIGHT = 8;
	static const int BOARD_WIDTH = 7;

	GameBoard(Player& player1, Player& player2);
	~GameBoard();

	// Prints the gameboard to the console
	void print() const;
	// Clears the board of pieces
	void clear();

	// Makes a move on the board, is retractible
	bool makeMove(const Move& move);
	// Retracts a move on the board
	bool retractMove();
	// Confirms a move to the board. This move is protected from retraction
	void confirmMove(const Move& move);
	// Returns the number of moves that have been confirmed
	inline unsigned int getConfirmedMovesCount() const { return confirmedMoves; }

	// Gets a piece at an (x,y) position. Both 0 based
	GamePiece* getPieceAt(int x, int y);
	// Gets a piece at a position
	inline GamePiece* getPieceAt(Position p) { return getPieceAt(p.getX(), p.getY()); }

	// Returns if a player has won
	bool hasEnded() const { return gameEnd; }

	// Sets the winner and the loser of the game
	void setWinner(Player* winner) { this->winner = winner; gameEnd = true; }
	// Sets the loser and the winner of the game
	void setLoser(Player* loser) { winner = (loser == player1 ? player2 : player1); gameEnd = true; }
	// Gets the winner
	inline Player* getWinner() const { return winner; }

	// Returns a pointer to player one
	Player* getPlayer1() const { return player1; }
	// Returns a pointer to player 2
	Player* getPlayer2() const { return player2; }
	// Returns a pointer that is the opposing player of the passed in.
	// If player is not from this board, returns NULL
	Player* getOtherPlayer(Player* player) const;

private:
	// Forward direction for pieces moving up the baord
	static const int UP = 1;
	// Forward direction for pieces moving down the board
	static const int DOWN = -UP;

	// The board state as a 2d vector of pointers to pieces
	std::vector<std::vector<GamePiece*> > board;
	// All of the pieces created wih this board. 
	std::vector<GamePiece*> pieces;

	// Pointer to the first player (goes up the board)
	Player* player1;
	// Pieces that player 1 owns
	std::unordered_map<int, GamePiece*>* player1Pieces;
	// Pointer to the second player (goes down the board)
	Player* player2;
	// Pieces that player 2 owns
	std::unordered_map<int, GamePiece*>* player2Pieces;

	// Number of confirmed moves
	unsigned int confirmedMoves;
	// Stack of the previous moves
	std::stack<Move> previousMoves;

	// If the game has ended
	bool gameEnd;
	// Winner of the game
	Player* winner;

	// Moves a piece from one square to another
	bool movePiece(Move move);

	// Foreground Color enum for printing the board on windows
	enum FGColor { WHITEFG = 7, RED = 12, GREEN = 2 };
	// Foreground Color enum for printing the baord on windows
	enum BGColor { BLACK = 0, WHITEBG = 7 };
	// Sets the output color of the console
	// Has ifdefs for windows and unix
	void setOutputColor(FGColor forground, BGColor background) const;
};
