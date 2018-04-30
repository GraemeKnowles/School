package prefixParser;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

// expression ::= "(", (function | operator) ,")"
class Expression extends Token {

	public static Expression parse(InputScanner input) {
		Expression e = new Expression();
		if (input.moveNext() == '(') {
			char peekNext = input.peekNext();
			if (OperatorName.isFirstCharValid(peekNext)) {
				e.function = BinaryOperator.parse(input);
			} else if (Function.isFirstCharValid(peekNext)) {
				e.function = Function.parse(input);
			} else {
				throw new IllegalArgumentException("Expression with no argument at index " + input.getIndex());
			}

			if (input.moveNext() == ')') {
				return e;
			} else {
				throw new IllegalArgumentException("Missing ) at index " + input.getIndex());
			}
		} else {
			throw new IllegalArgumentException("Missing ( at index " + input.getIndex());
		}
	}

	Function function;

	Expression() {
	}

	@Override
	public String toInfix() {
		return "(" + function.toInfix() + ")";
	}

	public static boolean isFirstCharValid(char c) {
		return c == '(';
	}

	@Override
	public List<Token> getSubTokens() {
		return new LinkedList<Token>(Arrays.asList(function));
	}
}
