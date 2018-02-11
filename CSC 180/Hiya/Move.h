#pragma once

#include "Position.h"

class GamePiece;

class Move
{
public:

	// from = position the move is moving a piece from
	// to = position to move the piece to 
	// attacking = piece the move is attacking
	// score = how good the move is
	Move(Position from = Position(0, 0), Position to = Position(0, 0), GamePiece* attacking = NULL, int score = 0);
	~Move() {}

	// Gets the location the move is moving from
	Position getFrom() const { return fromPos; }
	// Gets the location the move is moving to
	Position getTo() const { return toPos; }
	// Gets the piece the move is attacking
	GamePiece* getAttackedPiece() const { return attackedPiece; }
	// Sets the piece that is being attacked
	void setAttackedPiece(GamePiece* attackedPiece) { this->attackedPiece = attackedPiece; }
	int getScore() const { return score; }
	void setScore(int score) { this->score = score; }
	// Get the index into a history table for this move
	int getHistoryTableIndex() const { return historyTableIndex; }
	// Sets the weight pointer
	void setWeight(unsigned long* weightPtr) { this->weightPtr = weightPtr; }

	// Comparison operator(s)
	inline bool operator==(const Move& rhs) const { return (fromPos == rhs.fromPos && toPos == rhs.toPos); }
	// For sorting moves that are better than others
	bool operator<(const Move & rhs) const;

	// Returns the string representation of the move
	std::string toString() const;
	// Gets the mirror move (move from the other player's perspective)
	Move getTranslatedMove() const { return Move(fromPos.getMirrorPosition(), toPos.getMirrorPosition()); }

private:
	Position fromPos;
	Position toPos;
	int score;
	int historyTableIndex;
	GamePiece* attackedPiece;

	// Pointer to the 
	unsigned long* weightPtr;
	// 0 value to initialize the weightPtr to
	static unsigned long weight;
};