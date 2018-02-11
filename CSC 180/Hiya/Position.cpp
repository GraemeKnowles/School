#include "Position.h"
#include <algorithm>
#include <string>
#include "GameBoard.h"

std::string Position::intToAlpha[7] = { "A", "B", "C", "D", "E", "F", "G" };

Position::Position(std::string pos)
{
	std::transform(pos.begin(), pos.end(), pos.begin(), ::tolower);
	x = pos[0] - 'a';
	y = pos[1] - '0' - 1;
}

bool Position::isValid(std::string x)
{
	if (x.size() < 2) {
		return false;
	}

	int col = tolower(x[0]) - 'a';
	int row = x[1] - '0' - 1;

	if (col < 0 || row < 0 || col >= GameBoard::BOARD_WIDTH || row >= GameBoard::BOARD_HEIGHT)
	{
		return false;
	}

	return true;
}

Position Position::getMirrorPosition() const
{
	return Position(GameBoard::BOARD_WIDTH - x - 1, GameBoard::BOARD_HEIGHT - y - 1);
}