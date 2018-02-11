#include "AIPlayer.h"
#include "GameBoard.h"
#include "King.h"
#include "HumanPlayer.h"
#include "Move.h"
#include <climits>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <thread>
#include <future>
#include <queue>

using namespace std;

const unsigned AIPlayer::concurentThreadsSupported = std::thread::hardware_concurrency();
const int AIPlayer::MOVE_HISTORY_LENGTH = 7878;
mutex AIPlayer::statsMutex;
long long AIPlayer::staticMovesSearched;
long long AIPlayer::staticMovesPruned;

AIPlayer::AIPlayer(int turnTimeLimitMillis, int hardMaxDepthLimit)
	: timeLimitMillis(turnTimeLimitMillis), moveHistory(new unsigned long[MOVE_HISTORY_LENGTH]()), maxDepthLimit(hardMaxDepthLimit)
{

}

AIPlayer::~AIPlayer()
{
	delete[] moveHistory;

	for (unsigned int i = 0; i < selfCopies.size(); ++i)
	{
		delete selfCopies[i];
	}
	for (unsigned int i = 0; i < boardCopies.size(); ++i)
	{
		delete boardCopies[i];
	}
}

Move AIPlayer::getMove()
{
	// Performance statistics		
	staticMovesSearched = 0;
	staticMovesPruned = 0;
	// Start time to compare the runtime against fo the time limit
	startTime = getMillis();
	long long ellapsed = 0;
	// Start with worst possible best move
	Move best(Position(0, 0), Position(0, 0), NULL, INT_MIN);
	// Iterative Deepening for loop
	int maxDepth;
	for (maxDepth = 1; ellapsed < timeLimitMillis && maxDepth <= maxDepthLimit; ++maxDepth)
	{
		// Run the search at progressively deeper levels
		Move move = miniMax(maxDepth);
		ellapsed = (getMillis() - startTime);
		// If time limit expired
		if (ellapsed > timeLimitMillis)
		{
			break;
		}
		best = move;
	}
	//cout << "Moves Searched: " << staticMovesSearched << "\n";
	//cout << "Moves Pruned: " << staticMovesPruned << "\n";
	cout << maxDepth - 1 << " plies in " << ellapsed / 1000.0 << " seconds.\n";
	cout << "Computer Moves: " << best.toString() << " (" << best.getTranslatedMove().toString() << ")" << endl;
	return best;
}

// Creates a copy of the board for each child thread to use
void AIPlayer::createBoardCopies()
{
	if (getBoard() == NULL)
	{
		return;
	}

	for (unsigned int i = 0; i < concurentThreadsSupported; ++i)
	{
		AIPlayer* human = new AIPlayer(timeLimitMillis);
		selfCopies.push_back(human);
		AIPlayer* ai = new AIPlayer(timeLimitMillis);
		selfCopies.push_back(ai);
		GameBoard* board = new GameBoard(*human, *ai);
		boardCopies.push_back(board);
		boardPool.push(boardCopies[i]);
	}
}

// Updates the boards when a move is confirmed
void AIPlayer::updateBoardCopies(const Move & move)
{
	for (unsigned int i = 0; i < boardCopies.size(); ++i)
	{
		boardCopies[i]->confirmMove(translateMoveFromBoardToBoard(move, boardCopies[i]));
	}
}

// Since moves contain a pointer to the actual piece being attacked
// When getting the possible moves from the main board, and telling a 
// copy to make that move to be processed, the attacked piece in the move needs 
// to be changed to be the piece on the board copy. This also must be done when
// returning the best move
Move AIPlayer::translateMoveFromBoardToBoard(const Move & move, GameBoard * board) const
{
	Move translated(move.getFrom(), move.getTo());
	translated.setScore(move.getScore());
	GamePiece* attacked = move.getAttackedPiece();
	if (attacked != NULL)
	{
		translated.setAttackedPiece(board->getPieceAt(attacked->getPosition()));
	}
	return translated;
}

// MiniMax implementation to find the best move
// Includes Alpha Beta Pruning, Iterative Deepening, History Tables, Killer Move
// and is also multi-threaded.
Move AIPlayer::miniMax(int maxDepth)
{
	int depth = 1;
	Move best(Position(0, 0), Position(0, 0), NULL, INT_MIN);
	vector<Move> moves = getPossibleMoves();
	sortMoves(moves);

	// Boards that are in use
	queue<GameBoard*> runningBoards;
	// Moves corresponding to the board in the running board queue
	queue<Move> moveQueue;
	// Future values to be returned
	queue<future<int>> futures;

	//For each legal move m
	for (unsigned int i = 0; i < moves.size(); ++i)
	{
		// If no board is available to use
		if (boardPool.size() <= 0)
		{
			// Wait until a thread is done and process the value
			int moveValue = futures.front().get();
			futures.pop();
			if (moveValue > best.getScore())
			{
				moveQueue.front().setScore(moveValue);
				best = moveQueue.front();
			}
			// Remove the move
			moveQueue.pop();
			// retract the move
			GameBoard* board = runningBoards.front();
			board->retractMove();
			// remove the board from the run queue 
			runningBoards.pop();
			// and place it back in the pool
			boardPool.push(board);
		}

		// Get the next available board
		GameBoard* nextAvailableBoard = boardPool.front();
		// Remove it from the pool
		boardPool.pop();

		// Translate the move from the main board to the available board
		nextAvailableBoard->makeMove(translateMoveFromBoardToBoard(moves[i], nextAvailableBoard));
		moveQueue.push(moves[i]);

		// Support for playing against itself
		AIPlayer* player;
		if (getBoard()->getPlayer1() == this)
		{
			player = (AIPlayer*)nextAvailableBoard->getPlayer1();
		}
		else
		{
			player = (AIPlayer*)nextAvailableBoard->getPlayer2();
		}

		// Set up the ai player for this run
		player->maxDepth = maxDepth;
		player->startTime = startTime;
		// Set up statistics		
		player->movesPruned = 0;
		player->movesSearched = 0;
		// Spawn a thread that calculates the min for the move asynchronously 
		// Store it in a future to be accessed later
		futures.push(async([player, depth, best] {return player->min(depth, best); }));
		// put the board into the run queue
		runningBoards.push(nextAvailableBoard);
	}
	// Once all of the moves have been sent to be processed
	// Process the values of each move
	while (futures.size() > 0)
	{
		// Wait for the move to finish processing
		int moveValue = futures.front().get();
		// Remove it from the queue
		futures.pop();

		// Process the return value
		if (moveValue > best.getScore())
		{
			moveQueue.front().setScore(moveValue);
			best = moveQueue.front();
		}
		moveQueue.pop();

		// Retract the moves made. Only have to do the running boards because
		// those are the ones that have yet to be processed
		GameBoard* board = runningBoards.front();
		board->retractMove();
		// remove the board from the run queue 
		runningBoards.pop();
		// and place it back in the pool
		boardPool.push(board);
	}

	return translateMoveFromBoardToBoard(best, getBoard());
}

// This function is where the algorithm makes the opponent's moves.
// Its goal is to find the move that results in the least value
int AIPlayer::min(int depth, Move parentBest)
{
	vector<Move> moves = getOpponent()->getPossibleMoves();

	GameBoard* board = getBoard();

	if (moves.size() == 0)
	{
		board->setWinner(this);
	}
	if (board->hasEnded() || depth >= maxDepth)
	{
		return getBoardScore(depth, moves, true);
	}

	sortMoves(moves);

	Move best(Position(0, 0), Position(0, 0), NULL, INT_MAX);

	//for each human legal move m.mv
	auto end = moves.end();
	for (auto moveIt = moves.begin(); moveIt != end; ++moveIt)//, ++movesSearched)
	{
		//make move m.mv on Board
		board->makeMove(*moveIt);
		moveIt->setScore(max(depth + 1, best));
		//m.score = MAX
		if (moveIt->getScore() < best.getScore())
		{
			best = *moveIt;

			if (best.getScore() < parentBest.getScore())
			{
				board->retractMove();
				++(moveHistory[best.getHistoryTableIndex()]);
				killerMove = best;
				//movesPruned += distance(moveIt, end);
				break;
			}
		}
		// retract move m.mv on Board
		board->retractMove();

		if ((getMillis() - startTime) > timeLimitMillis)
		{
			break;
		}
	}

	//if (depth == 1)
	//{
	//	statsMutex.lock();
	//	staticMovesSearched += movesSearched;
	//	staticMovesPruned += movesPruned;
	//	statsMutex.unlock();
	//}

	//return best.score
	return best.getScore();
}

// This function is where the algorithm makes the AI's moves.
// Its goal is to find the move that results in the greatest value
int AIPlayer::max(int depth, const Move& parentBest)
{
	// Get the possible moves for this player
	vector<Move> moves = getPossibleMoves();

	GameBoard* board = getBoard();
	// Check to see if the player loses by running out of moves
	if (moves.size() == 0)
	{
		board->setWinner(getOpponent());
	}
	// Check if the search should stop
	if (board->hasEnded() || depth >= maxDepth)
	{
		return getBoardScore(depth, moves, false);
	}

	sortMoves(moves);

	// Initialize the best to the worst possible value
	Move best(Position(0, 0), Position(0, 0), NULL, INT_MIN);

	// For all moves
	auto end = moves.end();
	for (auto moveIt = moves.begin(); moveIt != moves.end(); ++moveIt)//, ++movesSearched)
	{
		// Make the move on the board
		board->makeMove(*moveIt);
		// Get the value of the opponents move
		moveIt->setScore(min(depth + 1, best));
		// If the value of the move is better than the best
		if (moveIt->getScore() > best.getScore())
		{
			// Set the new best
			best = *moveIt;

			// Check for alpha-beta cutoffs
			if (best.getScore() > parentBest.getScore())
			{
				// If the best score is worse for the parent (min) than its best
				// then we know the parent will not choose this move because it can only
				// get worse for the parent from here
				board->retractMove();
				// increment the move count in the history table
				++(moveHistory[best.getHistoryTableIndex()]);
				// Set the killer move
				killerMove = best;
				//movesPruned += distance(moveIt, end);
				break;
			}
		}

		// Return the board to its previous state
		board->retractMove();

		// Check if the time limit has been exceeded. The placement here means it will
		// only return values for a search that's reached the max depth
		if ((getMillis() - startTime) > timeLimitMillis)
		{
			break;
		}
	}

	//return best.score
	return best.getScore();
}

// Sorts moves for use with history tables and killer move
void AIPlayer::sortMoves(std::vector<Move>& possibleMoves) const
{
	unsigned long killerMoveWeight = UINT32_MAX;
	for (auto moveIt = possibleMoves.begin(); moveIt != possibleMoves.end(); ++moveIt)
	{
		// Check to see if any of the moves match the killer move,
		// set its weight to be sorted first
		if (*moveIt == killerMove)
		{
			moveIt->setWeight(&killerMoveWeight);
		}
		else
		{
			moveIt->setWeight(&(moveHistory[moveIt->getHistoryTableIndex()]));
		}
	}
	// Sort the moves to evaluate the most promising cutoffs first
	sort(possibleMoves.begin(), possibleMoves.end());
}

// Scores the current board state for a given depth
// possible moves is a list of either of the opponent's possible moves
// or the player's possible moves, specified by movesAreOpp.
// This is to save from having to re-calculate the possible moves for one of them
int AIPlayer::getBoardScore(int depth, const std::vector<Move>& possibleMoves, bool movesAreOpp) const
{
	// Cost of the end 
	static const int END = 1000000;
	int score = 0;
	// Check for board ending
	GameBoard* board = getBoard();
	if (board->hasEnded())
	{
		// Wins that happen earlier are better and losses that happen earlier are worse
		score += (board->getWinner() == this ? END : -END) / depth;
	}
	else
	{
		// Add up the scores of both players
		Player* opponent = getOpponent();
		if (movesAreOpp)
		{
			score += getScoreForPlayer(this, getPossibleMoves());
			score -= getScoreForPlayer(opponent, possibleMoves);
		}
		else
		{
			score += getScoreForPlayer(this, possibleMoves);
			score -= getScoreForPlayer(opponent, opponent->getPossibleMoves());
		}
	}

	return score;
}

// Get a score for the player. Input possible moves instead of generating to be able to
// Possibly re-use already generated moves (50% of the time)
int AIPlayer::getScoreForPlayer(const Player * player, const std::vector<Move>& possibleMoves) const
{
	// Health is the combined value of all the pieces
	static const int HEALTH_WEIGHT = 10;
	// Mobility is a measure of how many moves each player is able to make
	//static const int MOBILITY_WEIGHT = 1;	
	static const int POSS_ATTACK_VAL = 2;
	static const int POSS_MOVE_VAL = 1;
	static const int KING_THREAT_BONUS = 10;
	int health = 0, mobility = 0;

	// Get the player's health value
	for (auto piecesIt = player->getOwnedPieces()->begin(); piecesIt != player->getOwnedPieces()->end(); ++piecesIt)
	{
		health += piecesIt->second->getValue();
	}

	// get the player's mobility value
	for (auto piecesIt = possibleMoves.begin(); piecesIt != possibleMoves.end(); ++piecesIt)
	{
		GamePiece* attacked = piecesIt->getAttackedPiece();
		if (attacked != NULL)
		{
			// Bonus for being able to attack the king
			mobility += typeid(*attacked) == typeid(King) ? KING_THREAT_BONUS : POSS_ATTACK_VAL;
		}
		else
		{
			mobility += POSS_MOVE_VAL;
		}
	}

	return (HEALTH_WEIGHT * health) + mobility;
}

// Get the number of milliseconds from epoch
long long AIPlayer::getMillis() const
{
	return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())).count();
}
