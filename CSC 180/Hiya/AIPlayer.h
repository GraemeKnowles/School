#pragma once
#include "Player.h"
#include "HumanPlayer.h"
#include <climits>
#include <chrono>
#include <mutex>
#include <queue>

// The AI of HIYA! Contains implementation for minimax
class AIPlayer :
	public Player
{
public:
	// Time limit the AI should limit itself to and a
	// max depth iterative deepening should go to
	AIPlayer(int turnTimeLimitMillis, int hardMaxDepthLimit = INT_MAX);
	~AIPlayer();

	// Generates the move for the AI
	virtual Move getMove();

	// Creates copies of the boards for child threads to use
	void createBoardCopies();
	// Updates the boards when a move is confirmed
	void updateBoardCopies(const Move& move);
	
private:
	// Number of threads this system supports
	static const unsigned concurentThreadsSupported;

	// Current depth limit to search to
	int maxDepth;
	// Hard cap to stop iterative deepening at
	const int maxDepthLimit;
	
	// Time the search started
	long long startTime;
	// Time limit to end the search after
	unsigned long int timeLimitMillis;

	// Length of the move history table
	static const int MOVE_HISTORY_LENGTH;
	// Pointer to a move history table that is
	unsigned long * const moveHistory;
	// Move that last generated an alpha beta prune
	Move killerMove;

	// Virtual players to play on the boards of each thread
	std::vector<AIPlayer*> selfCopies;
	// Copies of the gameboard for each thread to use
	std::vector<GameBoard*> boardCopies;
	// The pool structure for the boards to be pulled from
	std::queue<GameBoard*> boardPool;

	// Since moves contain a pointer to the actual piece being attacked
	// When getting the possible moves from the main board, and telling a 
	// copy to make that move to be processed, the attacked piece in the move needs 
	// to be changed to be the piece on the board copy. This also must be done when
	// returning the best move
	Move translateMoveFromBoardToBoard(const Move& move, GameBoard* board) const;

	// MiniMax implementation to find the best move
	// Includes Alpha Beta Pruning, Iterative Deepening, History Tables, Killer Move
	// and is also multi-threaded.
	Move miniMax(int maxDepth);
	// Get the minimum move (opponent portion of minimax)
	int min(int depth, Move parentBest);
	// Get the maximum moves (self portion of minimax)
	int max(int depth, const Move& parentBest);
	// Sorts moves for use with history tables and killer move
	void sortMoves(std::vector<Move>& possibleMoves) const;
	
	// Scores the current board state for a given depth
	// possible moves is a list of either of the opponent's possible moves
	// or the player's possible moves, specified by movesAreOpp.
	// This is to save from having to re-calculate the possible moves for one of them
	int getBoardScore(int depth, const std::vector<Move>& possibleMoves, bool movesAreOpp) const;
	int getScoreForPlayer(const Player* player, const std::vector<Move>& possibleMoves) const;

	// Get the number of milliseconds from epoch
	long long getMillis() const;

	// Moves searched for a single thread		
	long long movesSearched;
	// Moves pruned for a single thread		
	long long movesPruned;
	// Mutex to control access to the statics variables shared between threads		
	static std::mutex statsMutex;
	// Total number of moves searched for all threads		
	static long long staticMovesSearched;
	// Total number of moves pruned for all threads		
	static long long staticMovesPruned;
};
