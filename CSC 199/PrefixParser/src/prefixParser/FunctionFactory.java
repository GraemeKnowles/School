package prefixParser;

public class FunctionFactory {

	public static Function createFunction(InputScanner input) {
		Function f = new Function();

		if (ValidName.isFirstCharValid(input.peekNext())) {
			f.setName(ValidName.parse(input));
		} else {
			throw new IllegalArgumentException("Invalid Function Name");
		}

		f.parseTerms(input);

		return f;
	}
}
