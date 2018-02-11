#pragma once

#include <string>
#include <vector>
#include "Move.h"

class GameBoard;
class Player;

// Base class for all game pieces of the game
class GamePiece
{
public:
	// gameboard - the board this piece is on
	// owner - the player that owns this piece
	// board representation - the character that represents this piece on the board
	// value - the value of this piece
	// demoteTo - if, when attacked, this piece demotes into another, the piece it demotes to
	GamePiece(GameBoard& gameBoard, Player& owner, std::string boardRep, int value, GamePiece* demoteTo);
	~GamePiece();

	// Returns the possible moves for this piece
	virtual std::vector<Move> getPossibleMoves() const = 0;
	// Returns the owner of this piece
	Player* getOwner() const { return owner; }
	// Gets the value of this piece
	int getValue() const { return value; }
	// Gets the string representing this piece
	std::string toString() const { return boardRepresentation; }
	// Returns the piece this demotes to
	virtual GamePiece* demoteTo() const { return demotePieceTo; }
	// Gets the current position of the piece
	Position getPosition() const { return Position(x, y); }
	// Gameboard is the creator and maintainer of the pieces
	// It needs access to the inner workings of the piece
	friend class GameBoard;
protected:
	// The board this piece is on
	GameBoard* gameBoard;
private:
	// x position of the piece
	int x;
	// y position of the piece
	int y;
	// value of the piece
	int value;
	// Player who owns the piece
	Player* owner;
	// string value that represents the piece
	std::string boardRepresentation;
	// piece this piece demotes into
	GamePiece* demotePieceTo;
	// Sets the position of the piece
	void setPosition(Position p) { x = p.getX(); y = p.getY(); }
	// Unique ID for the piece
	int uniqueID;
};
