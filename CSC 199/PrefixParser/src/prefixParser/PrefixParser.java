package prefixParser;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import prefixParser.Parser.Grapher;

public class PrefixParser {
	static final String helpShort = "-h";
	static final String helpLong = "--help";
	static final String verboseShort = "-v";
	static final String verboseLong = "--verbose";
	static final String wolframLong = "--wolfram";
	static final String desmosLong = "--desmos";
	static final String geogebraLong = "--geogebra";
	static final String examplesShort = "-e";
	static final String examplesLong = "--examples";
	static final String prefixParser = "PrefixParser";
	
	static class Arguments{
		public Grapher grapher;
		boolean verbose;
		String equation;
		boolean exitProgram;
		boolean runExample;
	}

	public static void main(String[] args) {		
		Arguments options = new Arguments();
		getOptions(args, options);
		
		if(options.runExample) {
			runExamples(options);
			return;
		}
		
		if(options.exitProgram) {
			return;
		}
		
		Parser p = new Parser(options.verbose, options.grapher);
		p.parse(options.equation);
		System.out.println(p.toInfix());
	}
	
	static void getOptions(String[] arguments, Arguments options) {
		options.equation = "";
		options.exitProgram = false;
		options.verbose = false;
		options.grapher = Grapher.WOLFRAM;
		
		List<String> argList = Arrays.asList(arguments);
		Iterator<String> argIterator = argList.iterator();
		
		// If no arguments, print help
		if (arguments.length == 0) {
			printHelp();
			options.exitProgram = true;
			return;
		}
		
		
		String firstArg = "";
		
		if(argIterator.hasNext()) {
			firstArg = argIterator.next();
		}
		
		String grapherArg = firstArg;
		
		// Check for help option
		if(arguments.length == 0 || firstArg.compareTo(helpShort) == 0 || firstArg.compareTo(helpLong) == 0) {
			printHelp();
			options.exitProgram = true;
			return;
		}
		
		if(firstArg.compareTo(examplesShort) == 0 || firstArg.compareTo(examplesLong) == 0) {
			options.runExample = true;
			if(argIterator.hasNext()) {
				grapherArg = argIterator.next();
			}
		}
		
		options.verbose = false;
		String verboseArg = null;
		// Get grapher option
		if(grapherArg.compareTo(wolframLong) == 0) {
			options.grapher = Grapher.WOLFRAM;
		}else if(grapherArg.compareTo(desmosLong) == 0) {
			options.grapher = Grapher.DESMOS;
		} else if (grapherArg.compareTo(geogebraLong) == 0) {
			options.grapher = Grapher.GEOGEBRA;
		}else {// If no grapher specified, use default
			options.grapher = Grapher.WOLFRAM;
			verboseArg = grapherArg;
		}
		
		// If the verbose arg isn't set, test for it
		boolean hasVerbose = false;
		if(verboseArg == null) {
			if(argIterator.hasNext()) {
				verboseArg = argIterator.next();
				hasVerbose = true;
			}
		}
		
		if(hasVerbose) {
			if(verboseArg.compareTo(verboseLong) == 0 || verboseArg.compareTo(verboseShort) == 0) {
				options.verbose = true;
			}else {// If the arg isn't verbose, assume it's part of the equation
				options.equation += verboseArg + " ";
			}
		}
		
		while(argIterator.hasNext()) {
			options.equation += argIterator.next() + " ";
		}
		
		options.exitProgram = false;
	}
	
	static void printHelp(){
		String[] helpText = {
				"Welcome to the fungp prefix parser. This program converts the prefix output of fungp and allows for output in infix.",
				"Infix is needed for many graphing calculators, many of which also have different support for various function names.",
				"Included in this program are options to tailor the output to work with various graphing calculators.",
				"\nUsage Syntax: ",
				prefixParser + " " + helpShort + " | " + helpLong,
				" " + helpShort + " " + helpLong + "       : Display help text that you're currently reading.",
				prefixParser + " " + "[" + wolframLong + " | " + desmosLong + " | " + geogebraLong + "] [" + verboseShort + " | " + verboseLong + "] equation",
				" " + wolframLong + "       : Configure for WolframAlpha, this is the default if not specified.",
				" " + desmosLong + "        : Configure for Desmos",
				" " + geogebraLong + "      : Configure for GeoGebra",
				" " + verboseShort + " " + verboseLong + "    : Output descriptive error information",
				prefixParser + " " + examplesShort + " | " + examplesLong + " [" + wolframLong + " | " + desmosLong + " | " + geogebraLong + "] [" + verboseShort + " | " + verboseLong + "]",
				" Run example equations with options"
		};
		
		for(String line : helpText) {
			System.out.println(line);
		}
	}
	
	static private void runExamples(Arguments options) {
		Parser p = new Parser(options.verbose, options.grapher);
		
		String[] tests = {
				"(+ (* (* c c) c) (+ (* c c) (+ (+ (+ c (- a (fungp.util/sdiv b 0.0))) c) c))))",
				"(+ (* (* c c) c) (+ (dec c) (+ (+ (+ c (- a (- b 0.0))) c) c))))", 
				"(mod (inc (dec (abs (gcd a b)))) a)",
				"(+ stuff b)",
				"(fungp.util/sdiv (Math/sin (fungp.util/sdiv (Math/sin 0.9) 0.0)) (inc (Math/sin (Math/sin x))))"
		};

		for(int i = 0; i < tests.length; ++i) {
			System.out.println("Test " + i + ": " + tests[i]);
			p.parse(tests[i]);
			System.out.println("Result: " + p.toInfix() + "\n");
		}
	}
}
