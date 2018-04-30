package prefixParser;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

//param ::= validName
class Parameter extends Token {

	public static Parameter parse(InputScanner input) {
		Parameter p = new Parameter();
		p.name = ValidName.parse(input);

		if (p.name.toString().compareTo("") == 0) {
			throw new IllegalArgumentException("Invalid Term at character index " + input.getIndex());
		}

		return p;
	}

	private ValidName name;

	Parameter() {
	}

	@Override
	public String toInfix() {
		return name.toString();
	}

	public static boolean isFirstCharValid(char c) {
		return ValidName.isFirstCharValid(c);
	}
	
	@Override
	public List<Token> getSubTokens() {
		return new LinkedList<Token>(Arrays.asList(name));
	}
	
	public ValidName getName() {
		return name;
	}
}
