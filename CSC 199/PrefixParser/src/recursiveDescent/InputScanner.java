package recursiveDescent;

class InputScanner {
	private String input;
	private int currentChar;
	
	public InputScanner(String input) {
		this.input = input;
		this.currentChar = -1;
	}
	
	public int getIndex() {
		return currentChar;
	}
	
	public char moveNext() {
		if(currentChar + 1 >= input.length()) {
			return 0;
		}
		return input.charAt(++currentChar);
	}
	
	public char peekNext() {
		if(currentChar + 1 >= input.length()) {
			return 0;
		}
		return input.charAt(currentChar + 1);
	}
}
