#pragma once

#include <string>

class Position
{
public:
	Position() {}
	Position(std::string pos);
	Position(int xPos, int yPos) : x(xPos), y(yPos) {}
	~Position() {}

	inline bool operator==(const Position& rhs) const { return (x == rhs.x && y == rhs.y); }

	std::string toString() const { return intToAlpha[x] + std::to_string(static_cast<long long>(y + 1)); }

	Position getMirrorPosition() const;

	int getX() const { return x; }
	int getY() const { return y; }

	static bool isValid(std::string x);

private:
	static std::string intToAlpha[7];
	int x;
	int y;
};
