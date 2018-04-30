package prefixParser;

import java.util.LinkedList;
import java.util.List;

// validName ::= validChar, {validChar}
// validChar ::= a-z | A-Z | 0-9 | _ | . | /
class ValidName extends Token {

	public static ValidName parse(InputScanner input) {
		ValidName fn = new ValidName();
		fn.setName(input);
		return fn;
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
		return Character.isJavaIdentifierPart(c) || c == '/' || c == '.' ;
	}
	
	@Override
	public List<Token> getSubTokens() {
		return new LinkedList<Token>();
	}
	
	public void setName(String name) {
		String oldName = this.name;
		setName(new InputScanner(name));
		if(name == "") {
			name = oldName;
		}
	}
	
	private void setName(InputScanner input) {
		this.name = "";
		do {
			char nextChar = input.peekNext();
			if (input.nextValid() && isFirstCharValid(nextChar)) {
				name += input.moveNext();
			} else {
				return;
			}
		} while (true);
	}
}
