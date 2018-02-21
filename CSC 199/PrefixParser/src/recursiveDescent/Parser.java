package recursiveDescent;

// This is a recursive descent parser based on the following Infix Grammar.

// expression ::= "(", (function | operator) ,")"
// function ::= validName, {" ", term}
// validName ::= validChar, {validChar}
// validChar ::= a-z | A-Z | 0-9 | _ | . | /
// term ::= param | expression
// operator ::= operatorName, " ", term, " ", term
// operatorName ::= + | - | / | * 
// param ::= validName

// All rules have their own class with a static function called parse(...) that parses each 
// rule into its own token that contains references to each sub-token that it is comprised of. 

// Because the output syntax for fungp uses parenthesis, order of operations
// need not be accounted for. Each token has an override of the function 
// toInfix() that returns the infix version of that token.

// Special output cases like the safe divide are currently handled in the toInfix() override of the Function class.

public class Parser {

	Token parsed = null;
	
	// Determines if the parser should output error messages to the console
	boolean verboseErrors = false;
	
	public Parser() {}
	
	public Parser(String input) {
		this(input, false);
	}
	
	public Parser(boolean verboseErrors) {
		this.verboseErrors = verboseErrors;
	}
	
	public Parser(String input, boolean verboseErrors) {
		this.verboseErrors = verboseErrors;
		parse(input);
	}

	// Parses input prefix string.
	public boolean parse(String input) {
		parsed = null;
		try {
			parsed = Expression.parse(new InputScanner(input));
		} catch (IllegalArgumentException iae) {
			if (verboseErrors) {
				System.out.println(iae.getMessage());
			}
			return false;
		}
		return true;
	}

	// Outputs previously parsed input string in infix notation
	public String toInfix() {
		if (parsed == null) {
			return "";
		}
		return parsed.toInfix();
	}
	
	public boolean isVerboseErrors() {
		return verboseErrors;
	}

	public void setVerboseErrors(boolean verboseErrors) {
		this.verboseErrors = verboseErrors;
	}

}
