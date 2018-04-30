package prefixParser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

// function ::= validName, {" ", term}
class Function extends Token {

	public static Function parse(InputScanner input) {
		Function f = new Function();

		if (ValidName.isFirstCharValid(input.peekNext())) {
			f.name = ValidName.parse(input);
		} else {
			throw new IllegalArgumentException("Invalid Function Name");
		}

		f.parseTerms(input);

		return f;
	}

	private ValidName name;
	private List<Term> terms = new ArrayList<Term>();

	Function() {
	}

	protected void parseTerms(InputScanner input) {
		do {
			char nextChar = input.peekNext();
			if (nextChar == ' ') {
				input.moveNext();
				if (Term.isFirstCharValid(input.peekNext())) {
					this.addTerm(Term.parse(input));
				} else {
					return;
				}
			} else {
				return;
			}
		} while (true);
	}

	protected List<Term> getTerms() {
		return terms;
	}

	protected void addTerm(Term t) {
		terms.add(t);
	}

	protected void setName(ValidName name) {
		this.name = name;
	}

	public String getName() {
		return name.toString();
	}

	@Override
	public String toInfix() {

		final String fungpUtil = "fungp.util/";
		final String math = "Math/";

		// Clean name
		String name = getName();
		if (name.contains(fungpUtil)) {
			name = name.substring(fungpUtil.length());
		} else if (name.contains(math)) {
			name = name.substring(math.length());
		}

		// Handle special cases
		switch (name) {
		case "sdiv":
			String denom = terms.get(1).toInfix();
			try {
				if(Float.parseFloat(denom) == 0) {
					denom = "1";
				}
			} catch (NumberFormatException ex) {
				// Denom is not a constant, do nothing, just print it.
			}
			return terms.get(0).toInfix() + "/" + denom;
		// break;
		case "inc":
			return terms.get(0).toInfix() + "+1";
		// break;
		case "dec":
			return terms.get(0).toInfix() + "-1";
		// break;
		case "expt":
			return terms.get(0).toInfix() + "^" + terms.get(1).toInfix();
		// break;
		case "rem":
			name = "mod";
		// fallthrough intended
		default:
			String infix = name + "(";
			for (int i = 0, lastComma = terms.size() - 1; i < terms.size(); ++i) {
				infix += terms.get(i).toInfix();
				if (i < lastComma) {
					infix += ",";
				}
			}

			return infix + ")";
		// break;
		}
	}

	public static boolean isFirstCharValid(char c) {
		return ValidName.isFirstCharValid(c);
	}
	
	@Override
	public List<Token> getSubTokens() {
		return new LinkedList<Token>(terms);
	}
}
