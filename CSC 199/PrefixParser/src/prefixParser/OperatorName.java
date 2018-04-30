package prefixParser;

// operatorName ::= + | - | / | * 
class OperatorName extends ValidName {

	public static OperatorName parse(InputScanner input) {
		OperatorName fn = new OperatorName();

		do {
			char nextChar = input.peekNext();
			if (isFirstCharValid(nextChar)) {
				fn.name += input.moveNext();
			} else {
				return fn;
			}
		} while (true);
	}

	OperatorName() {
	}

	public static boolean isFirstCharValid(char c) {
		return c == '+' || c == '-' || c == '/' || c == '*';
	}
}
