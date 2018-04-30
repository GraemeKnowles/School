package prefixParser;

import java.util.List;

abstract class Token{
	// Returns the infix notation of the token
	public abstract String toInfix();
	
	public abstract List<Token> getSubTokens();
}
