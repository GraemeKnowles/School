package recursiveDescent;

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
}
