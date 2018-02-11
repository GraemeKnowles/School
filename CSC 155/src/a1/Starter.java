package a1;

/**
 * The main entry point into Assignment 1.
 * Creates and maintains the MVC architecture.
 * 
 * @author Graeme Knowles
 */
public class Starter extends javax.swing.JFrame {
	
	/**
	 * The triangle model being acted on
	 */
	private static TriangleModel model = null;
	
	
	/**
	 * The view observing the model
	 */
	private static TriangleView view = null;
	
	
	/**
	 * The controller acting on the model
	 */
	private static TriangleController controller = null;
	
	public static void main(String[] args) {	
		// Print out the java version
		System.out.println("Java Version: " + System.getProperty("java.version"));
		
		// Instantiate and link the MVC architecture together
		model = new TriangleModel();
		view = new TriangleView(new ModelProxy(model));
		controller = new TriangleController(model, view);
	}

}
