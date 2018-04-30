package prefixParser;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

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

	enum Grapher{WOLFRAM, DESMOS, GEOGEBRA};
	Grapher grapher = null;

	Token parsed = null;
	
	// Determines if the parser should output error messages to the console
	boolean verboseErrors = false;
	
	public Parser(boolean verboseErrors, Grapher grapher)
	{
		this.verboseErrors = verboseErrors;
		this.grapher = grapher;
	}
	
	public Parser(String input, boolean verboseErrors, Grapher grapher) {
		this(verboseErrors, grapher);
		parse(input);
	}

	// Parses input prefix string.
	public boolean parse(String input) {
		parsed = null;
		try {
			parsed = Expression.parse(new InputScanner(input));
			outputTokenPass(parsed);
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
		
		return this.outputStringPass(parsed.toInfix());
	}
	
	public Grapher getGrapher() {
		return grapher;
	}

	public void setGrapher(Grapher grapher) {
		this.grapher = grapher;
	}
	
	public boolean isVerboseErrors() {
		return verboseErrors;
	}

	public void setVerboseErrors(boolean verboseErrors) {
		this.verboseErrors = verboseErrors;
	}
	
	private void outputTokenPass(Token parsed) {
		switch(grapher) {
		case WOLFRAM:
			wolfram(parsed);
			break;
		case DESMOS:
			desmos(parsed);
			break;
		case GEOGEBRA:
			geogebra(parsed);
			break;
		}
	}
	
	private String outputStringPass(String infix) {
		String output = infix;
		
		switch(grapher) {
		case WOLFRAM:
			output = "z = " + output;
			break;
		case DESMOS:
			output = "f(x,y) = " + output;
			break;
		case GEOGEBRA:
			output = "z = " + output;
			break;
		}
		
		return output;
	}
	
	private void wolfram(Token parsed) {
		
	}
	
	private void desmos(Token parsed) {
		xyConverter(parsed);
	}
	
	private void geogebra(Token parsed) {
		xyConverter(parsed);
	}

	//converts parameters from whatever they were named before to x and y
	private void xyConverter(Token parsed) {
		// Get all parameters
		Queue<Token> tokens = new LinkedList<Token>();
		List<Parameter> parameters = new LinkedList<Parameter>();
		tokens.add(parsed);
		for(Token current = tokens.poll(); current != null; current = tokens.poll()){
			
			if(current instanceof Parameter) {
				// if not a number
				if(!((Parameter)current).getName().toString().matches("-?\\d+(\\.\\d+)?")) {
					parameters.add((Parameter) current);
				}
			}
			tokens.addAll(current.getSubTokens());
		}
		
		// Replace parameters with x and y
		List<String> paramNames = new LinkedList<String>();
		List<LinkedList<Parameter>> params = new ArrayList<LinkedList<Parameter>>();
		
		for(int i = 0; i < parameters.size(); ++i) {
			Parameter currentParam = parameters.get(i);
			String currentParamName = currentParam.getName().toString();
			// Compare to stored parameter names
			int index = -1;
			for(int j = 0; j < paramNames.size(); ++j) {
				if(paramNames.get(j).compareTo(currentParamName) == 0) {
					index = j;
					break;
				}
			}
			
			// If it doesn't exist, create a new list
			if(index == -1) {
				paramNames.add(currentParamName);
				params.add(new LinkedList<Parameter>());
				params.get(params.size() - 1).add(currentParam);
			}else {// else add it to the existing list
				params.get(index).add(currentParam);
			}
		}
		
		String[] paramsToSwitchTo = {"x", "y"};
		
		if(params.size() > paramsToSwitchTo.length) {
			// Number of parameters too long to display in 3d.
			// throw error?
			return;
		}
		
		for(int i = 0; i < params.size(); ++i) {
			List<Parameter> currentParamList = params.get(i);
			for(int j = 0; j < currentParamList.size(); ++j) {
				currentParamList.get(j).getName().setName(paramsToSwitchTo[i]);
			}
		}
	}
}
