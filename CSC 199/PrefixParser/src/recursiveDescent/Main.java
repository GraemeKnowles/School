package recursiveDescent;

public class Main {

	public static void main(String[] args) {
		String[] tests = {"(+ (* (* c c) c) (+ (* c c) (+ (+ (+ c (- a (fungp.util/sdiv b 0.0))) c) c))))",
				"(+ (* (* c c) c) (+ (dec c) (+ (+ (+ c (- a (- b 0.0))) c) c))))", 
				"(mod (inc (dec (abs (gcd x y)))) x)"};
		
		Parser p = new Parser(true);
		
		for(int i = 0; i < tests.length; ++i) {
			System.out.println("Test " + i + ": " + tests[i]);
			p.parse(tests[i]);
			System.out.println("Result: " + p.toInfix() + "\n");
		}
	}
}
