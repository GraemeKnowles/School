package prefixParser;

import java.util.List;

// operator ::= operatorName, " ", term, " ", term
// operatorName ::= + | - | / | * 
class BinaryOperator extends Function {

	public static BinaryOperator parse(InputScanner input) {
		BinaryOperator op = new BinaryOperator();

		char nextChar = input.peekNext();
		if (OperatorName.isFirstCharValid(nextChar)) {
			op.setName(OperatorName.parse(input));
		} else {
			throw new IllegalArgumentException("Invalid Operator Name");
		}

		int index = input.getIndex();// Get index of operator to use for more descriptive error

		op.parseTerms(input);
		if (op.getTerms().size() != 2) {
			throw new IllegalArgumentException("Expecting two parameters for operator at index " + index);
		}

		return op;
	}

	BinaryOperator() {
	}

	@Override
	public String toInfix() {
		List<Term> terms = this.getTerms();
		String infix = terms.get(0).toInfix();
		infix += this.getName();
		infix += terms.get(1).toInfix();
		return infix;
	}

	public static boolean isFirstCharValid(char c) {
		return OperatorName.isFirstCharValid(c);
	}
}
