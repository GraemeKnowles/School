#include "GameBoard.h"
#include "King.h"
#include "Ninja.h"
#include "MiniNinja.h"
#include "Samurai.h"
#include "MiniSamurai.h"
#include "Player.h"
#include <iostream>
#include <string>
#include <typeinfo>

#ifdef _WIN32
#include <windows.h>   // WinApi header
#endif

GameBoard::GameBoard(Player& player1, Player& player2) : confirmedMoves(0), player1Pieces(new std::unordered_map<int, GamePiece*>()), player2Pieces(new std::unordered_map<int, GamePiece*>())
{
	// Associate the players with this board
	this->player1 = &player1;
	player1.attackingDirection = UP;
	player1.ownedPieces = player1Pieces;
	player1.setBoard(*this);
	this->player2 = &player2;
	player2.ownedPieces = player2Pieces;
	player2.attackingDirection = DOWN;
	player2.setBoard(*this);
	// Set the players versus each other
	player1.setOpponent(player2);
	player2.setOpponent(player1);

	// Set up the initial board state
	gameEnd = false;
	winner = NULL;

	// Set the board to the right size
	board.resize(BOARD_WIDTH);
	for (unsigned int i = 0; i < board.size(); ++i) 
	{
		board[i].resize(BOARD_HEIGHT);
	}
	// Number of each type of piece
	static const int PIECE_COUNT = 3;
	// Constants for the big and small pieces
	static const int BIG_ROW = 1;
	static const int MIN_ROW = 2;
	static const int COMP_BIG_ROW = BOARD_HEIGHT - BIG_ROW - 1;
	static const int COMP_MIN_ROW = BOARD_HEIGHT - MIN_ROW - 1;

	// Constants for player1s pieces
	static const int NINJA_IND = 4;
	static const int MIN_NINJA_IND = 0;
	static const int SAM_IND = 0;
	static const int MIN_SAM_IND = 4;
	// Constants for player2s pieces
	static const int COMP_NINJA_IND = 0;
	static const int COMP_MIN_NINJA_IND = 4;
	static const int COMP_SAM_IND = 4;
	static const int COMP_MIN_SAM_IND = 0;
	// Constants for the kings
	static const int KING_ROW = 0;
	static const int KING_IND = 3;
	// Place Kings
	GamePiece* piece = new King(*this, player1);
	(*player1Pieces)[piece->uniqueID] = piece;
	board[KING_IND][KING_ROW] = piece;
	piece->setPosition(Position(KING_IND, KING_ROW));
	pieces.push_back(piece);

	piece = new King(*this, player2);
	(*player2Pieces)[piece->uniqueID] = piece;
	board[KING_IND][BOARD_HEIGHT - KING_ROW - 1] = piece;
	piece->setPosition(Position(KING_IND, BOARD_HEIGHT - KING_ROW - 1));
	pieces.push_back(piece);

	// Place pieces
	for (int i = 0; i < PIECE_COUNT; ++i) {
		int x = 0, y = 0;
		// Human Pieces
		piece = new Ninja(*this, player1);
		(*player1Pieces)[piece->uniqueID] = piece;
		board[x = NINJA_IND + i][y = BIG_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new MiniNinja(*this, player1);
		(*player1Pieces)[piece->uniqueID] = piece;
		board[x = MIN_NINJA_IND + i][y = MIN_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new Samurai(*this, player1);
		(*player1Pieces)[piece->uniqueID] = piece;
		board[x = SAM_IND + i][y = BIG_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new MiniSamurai(*this, player1);
		(*player1Pieces)[piece->uniqueID] = piece;
		board[x = MIN_SAM_IND + i][y = MIN_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		// Computer Pieces
		piece = new Ninja(*this, player2);
		(*player2Pieces)[piece->uniqueID] = piece;
		board[x = COMP_NINJA_IND + i][y = COMP_BIG_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new MiniNinja(*this, player2);
		(*player2Pieces)[piece->uniqueID] = piece;
		board[x = COMP_MIN_NINJA_IND + i][y = COMP_MIN_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new Samurai(*this, player2);
		(*player2Pieces)[piece->uniqueID] = piece;
		board[x = COMP_SAM_IND + i][y = COMP_BIG_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);

		piece = new MiniSamurai(*this, player2);
		(*player2Pieces)[piece->uniqueID] = piece;
		board[x = COMP_MIN_SAM_IND + i][y = COMP_MIN_ROW] = piece;
		piece->setPosition(Position(x, y));
		pieces.push_back(piece);
	}
}

GameBoard::~GameBoard()
{
	clear();
}

// Prints the gameboard to the console
void GameBoard::print() const
{
	setOutputColor(WHITEFG, BLACK);
	std::cout << "      ---------------------   Computer    |    ---------------------   Human" << std::endl;

	for (int i = BOARD_HEIGHT - 1; i >= 0; --i)
	{
		int row = i + 1;
		setOutputColor(WHITEFG, BLACK);
		std::cout << "    " << row << " ";
		int colorMod = 0;
		if (i % 2 != 0) {
			colorMod = 1;
		}

		for (int x = 0; x < 2; ++x)
		{
			if (x == 1)
			{
				// Rotated board
				setOutputColor(WHITEFG, BLACK);
				std::cout << "               |  " << row << " ";
			}
			for (int j = 0; j < BOARD_WIDTH; ++j)
			{
				BGColor bgVal = ((j + colorMod + x) % 2 == 0 ? BLACK : WHITEBG);
				std::string out = "   ";
				GamePiece* piece;
				if (0 == x)
				{
					piece = board[j][i];
				}
				else
				{
					Position rotatedPosition = (Position(j, i)).getMirrorPosition();
					piece = board[rotatedPosition.getX()][rotatedPosition.getY()];
				}

				FGColor fgVal = WHITEFG;
				if (piece != NULL)
				{
					fgVal = piece->getOwner() == player1 ? GREEN : RED;
					out = " " + piece->toString() + " ";
				}
				setOutputColor(fgVal, bgVal);
				std::cout << out;
			}
		}
		setOutputColor(WHITEFG, BLACK);

		std::cout << std::endl;
	}

	setOutputColor(WHITEFG, BLACK);
	std::cout << "      ---------------------   Human       |    ---------------------   Computer\n";
	std::cout << "       A  B  C  D  E  F  G                |     A  B  C  D  E  F  G" << std::endl;
}

// Clears the board pieces
void GameBoard::clear()
{
	for (unsigned int i = 0; i < pieces.size(); ++i)
	{
		delete pieces[i];
	}

	board.clear();

	if (player2Pieces != NULL)
	{
		player2Pieces->clear();
		delete player2Pieces;
	}

	if (player1Pieces != NULL)
	{
		player1Pieces->clear();
		delete player1Pieces;
	}
}

// Makes a move on the board, is retractible
bool GameBoard::makeMove(const Move& move)
{
	// Move the piece
	GamePiece* pieceToMove = getPieceAt(move.getFrom());
	if (!movePiece(move))
	{
		return false;
	}

	// Check for a target
	GamePiece* targetPiece = move.getAttackedPiece();
	if (targetPiece != NULL)
	{
		Player* attackedPieceOwner = targetPiece->getOwner();

		// Prevent pieces from attacking their own kind
		if (attackedPieceOwner != pieceToMove->getOwner())
		{
			// Check if a king is attacked
			if (typeid(*targetPiece) == typeid(King)) {
				gameEnd = true;
				winner = pieceToMove->getOwner();
			}

			std::unordered_map<int, GamePiece*>* ownerPieces = attackedPieceOwner->getOwnedPieces();
			// Remove the target from the list of pieces the owner owns
			ownerPieces->erase(targetPiece->uniqueID);
			// Replace the target with whatever it demotes into
			GamePiece* targetDemotesTo = targetPiece->demoteTo();
			// Place the demoted piece
			Position p = targetPiece->getPosition();
			if (targetDemotesTo != NULL)
			{
				(*ownerPieces)[targetDemotesTo->uniqueID] = targetDemotesTo;
				targetDemotesTo->setPosition(p);
			}
			board[p.getX()][p.getY()] = targetDemotesTo;
		}
	}

	// Add the move to the list of moves
	previousMoves.push(move);

	return true;
}

// Confirms a move to the board. This move is protected from retraction
bool GameBoard::retractMove()
{
	// Check to see if there are moves to retract
	if (previousMoves.size() <= confirmedMoves)
	{
		return false;
	}

	gameEnd = false;
	this->winner = NULL;

	// Get the previous move
	Move move = previousMoves.top();
	previousMoves.pop();

	// If the move attacked a piece, replace the piece it demoted into
	GamePiece* attackedPiece = move.getAttackedPiece();
	if (attackedPiece != NULL)
	{
		Player* owner = attackedPiece->getOwner();
		auto ownedPieces = owner->getOwnedPieces();
		// Remove the demoted piece
		GamePiece* demotedTo = attackedPiece->demoteTo();
		Position p = attackedPiece->getPosition();
		if (demotedTo != NULL)
		{
			ownedPieces->erase(demotedTo->uniqueID);
			p = demotedTo->getPosition();
		}
		// Add the attacked piece
		(*ownedPieces)[attackedPiece->uniqueID] = attackedPiece;
		// Place the attacked piece back on the board
		board[p.getX()][p.getY()] = attackedPiece;
	}

	// Move the piece back to its previous position
	movePiece(Move(move.getTo(), move.getFrom()));

	return true;
}

// Gets a piece at an (x,y) position. Both 0 based
GamePiece * GameBoard::getPieceAt(int x, int y)
{
	if (y >= BOARD_HEIGHT || y < 0 || x >= BOARD_WIDTH || x < 0)
	{
		return NULL;
	}

	return board[x][y];
}

// Returns a pointer that is the opposing player of the passed in.
// If player is not from this board, returns NULL
Player * GameBoard::getOtherPlayer(Player * player) const
{
	if (player == player1)
	{
		return player2;
	}
	else if (player == player2)
	{
		return player1;

	}
	return NULL;
}

void GameBoard::confirmMove(const Move& move)
{
	// Clear any unconfirmed moves
	for (; confirmedMoves < previousMoves.size(); previousMoves.pop());

	// Sanitize the move
	Move moveCopy(move.getFrom(), move.getTo());
	GamePiece* attacked = move.getAttackedPiece();
	if (attacked != NULL)
	{
		moveCopy.setAttackedPiece(getPieceAt(attacked->getPosition()));
	}
	// Make the move
	makeMove(move);
	++confirmedMoves;
}

// Moves a piece from one square to another
bool GameBoard::movePiece(Move move)
{
	// Only move to an empty position
	Position to = move.getTo();
	int xTo = to.getX();
	int yTo = to.getY();
	if (board[xTo][yTo] != NULL) {
		return false;
	}

	// Check to make sure there's a piece to move
	Position from = move.getFrom();
	int xFrom = from.getX();
	int yFrom = from.getY();
	if (board[xFrom][yFrom] == NULL) {
		return false;
	}

	board[xFrom][yFrom]->setPosition(to);
	board[xTo][yTo] = board[xFrom][yFrom];
	board[xFrom][yFrom] = NULL;
	return true;
}

// Sets the output color of the console
void GameBoard::setOutputColor(FGColor forground, BGColor background) const
{
#ifdef _WIN32
	static const HANDLE  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	static const int WHITE_BG_OFF = 112;
	int color = forground;

	switch (background)
	{
	case BLACK:
		break;
	case WHITEBG:
		color += WHITE_BG_OFF;
		break;
	}

	SetConsoleTextAttribute(hConsole, color);

#elif __unix__
	static const int ESC_SIZE = 6;
	static const char BLACK_BACK[ESC_SIZE] = { '\033', '[', '4', '0' , 'm', '\0' };
	static const char WHITE_BACK[ESC_SIZE] = { '\033', '[', '4', '7', 'm', '\0' };
	static const char WHITE_FOR[ESC_SIZE] = { '\033', '[', '3', '7' , 'm', '\0' };
	static const char RED_FOR[ESC_SIZE] = { '\033', '[', '3', '1', 'm', '\0' };
	static const char GREEN_FOR[ESC_SIZE] = { '\033', '[', '3', '2', 'm', '\0' };

	switch (forground)
	{
	case GREEN:
		std::cout << GREEN_FOR;
		break;
	case RED:
		std::cout << RED_FOR;
		break;
	case WHITEFG:
		std::cout << WHITE_FOR;
	}

	switch (background)
	{
	case BLACK:
		std::cout << BLACK_BACK;
		break;
	case WHITEBG:
		std::cout << WHITE_BACK;
		break;
	}
#endif
}
