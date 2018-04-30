package prefixParser;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

// term ::= param | expression
class Term extends Token {

	public static Term parse(InputScanner input) {
		Term t = new Term();

		char nextChar = input.peekNext();
		if (Expression.isFirstCharValid(nextChar)) {
			t.paramOrExpression = Expression.parse(input);
		} else if (Parameter.isFirstCharValid(nextChar)) {
			t.paramOrExpression = Parameter.parse(input);
		}

		return t;
	}

	private Token paramOrExpression;

	Term() {
	}

	@Override
	public String toInfix() {
		return paramOrExpression.toInfix();
	}

	public static boolean isFirstCharValid(char c) {
		return Expression.isFirstCharValid(c) || ValidName.isFirstCharValid(c);
	}
	
	@Override
	public List<Token> getSubTokens() {
		return new LinkedList<Token>(Arrays.asList(paramOrExpression));
	}
}
