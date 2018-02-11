package a2;

/**
 * The main entry point into Assignment 1. Creates and maintains the MVC
 * architecture.
 *
 * @author Graeme Knowles
 */
public class Starter extends javax.swing.JFrame {

	/**
	 * The triangle model being acted on
	 */
	private static SolarSystem model = null;

	/**
	 * The view observing the model
	 */
	private static View view = null;

	/**
	 * The controller acting on the model
	 */
	private static Controller controller = null;

	public static void main(String[] args) {
		// Instantiate and link the MVC architecture together
		model = new SolarSystem();
		view = new View(model, new PerspectiveCamera());
		controller = new Controller(model, view);
	}

}
