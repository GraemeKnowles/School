package recursiveDescent;

// validName ::= validChar, {validChar}
// validChar ::= a-z | A-Z | 0-9 | _ | . | /
class ValidName extends Token {

	public static ValidName parse(InputScanner input) {
		ValidName fn = new ValidName();

		do {
			char nextChar = input.peekNext();
			if (isFirstCharValid(nextChar)) {
				fn.name += input.moveNext();
			} else {
				return fn;
			}
		} while (true);
	}

	protected String name;

	ValidName() {
		name = "";
	}

	@Override
	public String toString() {
		return name;
	}

	@Override
	public String toInfix() {
		return name;
	}

	public static boolean isFirstCharValid(char c) {
		return Character.isJavaIdentifierPart(c) || c == '/' || c == '.';
	}
}