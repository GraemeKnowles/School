package a2;

import java.awt.BorderLayout;

import javax.swing.ActionMap;
import javax.swing.InputMap;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.KeyStroke;

import com.jogamp.opengl.util.FPSAnimator;

/**
 * This class is the controller portion of the Model-View-Controller
 * architecture It acts upon a model (or creates objects that act upon it)
 *
 * @author Graeme Knowles
 */
public class Controller extends JFrame {

	/**
	 * The model that contains all the information being displayed
	 */
	private SolarSystem model = null;

	// GUI

	/**
	 * The center panel of the window. This contains the OpenGL canvas
	 */
	private JPanel centerPanel = null;

	/**
	 * The View portion of the MVC architecture. This is the openGL canvas where the
	 * Objects are drawn
	 */
	private View canvas = null;

	/**
	 * The object the drives the animation
	 */
	FPSAnimator animator = null;

	/**
	 * @param model
	 *            The model the controller acts on
	 * @param view
	 *            The view that displays the information from the model
	 */
	public Controller(SolarSystem model, View view) {
		this.model = model;
		this.canvas = view;
		initGUI();
	}

	/**
	 * Initializes the graphical user interface
	 */
	private void initGUI() {
		// Set the settings on the program window
		setTitle("Graeme Knowles - Assignment 1");
		setSize(800, 800);
		setLocation(200, 200);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		// Create + Add the objects
		BorderLayout centerPanelLayout = new BorderLayout();
		centerPanel = new JPanel(centerPanelLayout);
		// add the drawing canvas
		centerPanel.add(canvas);
		this.add(centerPanel, BorderLayout.CENTER);

		// a key binding that toggles the triangle between a single solid color,
		// and a gradient of three colors when user presses the ‘c’ key.
		// get the "focus is in the window" input map for the center panel
		int mapName = JComponent.WHEN_IN_FOCUSED_WINDOW;
		InputMap imap = centerPanel.getInputMap(mapName);
		// create a keystroke object to represent the "c" key
		KeyStroke wKey = KeyStroke.getKeyStroke('w');
		KeyStroke aKey = KeyStroke.getKeyStroke('a');
		KeyStroke sKey = KeyStroke.getKeyStroke('s');
		KeyStroke dKey = KeyStroke.getKeyStroke('d');
		KeyStroke eKey = KeyStroke.getKeyStroke('e');
		KeyStroke qKey = KeyStroke.getKeyStroke('q');
		KeyStroke zKey = KeyStroke.getKeyStroke('z');
		KeyStroke cKey = KeyStroke.getKeyStroke('c');
		KeyStroke rKey = KeyStroke.getKeyStroke('r');
		KeyStroke fKey = KeyStroke.getKeyStroke('f');
		KeyStroke up = KeyStroke.getKeyStroke("UP");
		KeyStroke down = KeyStroke.getKeyStroke("DOWN");
		KeyStroke left = KeyStroke.getKeyStroke("LEFT");
		KeyStroke right = KeyStroke.getKeyStroke("RIGHT");
		KeyStroke space = KeyStroke.getKeyStroke("SPACE");

		// put the "cKey" keystroke object into the central panel’s "when focus is
		// in the window" input map under the identifier name "color“
		String forwardKey = "forward";
		imap.put(wKey, forwardKey);
		String backwardKey = "backward";
		imap.put(sKey, backwardKey);
		String leftKey = "left";
		imap.put(aKey, leftKey);
		String rightKey = "right";
		imap.put(dKey, rightKey);
		String upKey = "up";
		imap.put(qKey, upKey);
		String downKey = "down";
		imap.put(eKey, downKey);
		String rotateDown = "rdown";
		imap.put(down, rotateDown);
		String rotateUp = "rup";
		imap.put(up, rotateUp);
		String rotateLeft = "rleft";
		imap.put(left, rotateLeft);
		String rotateRight = "rright";
		imap.put(right, rotateRight);
		String rollRight = "rlright";
		imap.put(zKey, rollRight);
		String rollLeft = "rlleft";
		imap.put(cKey, rollLeft);
		String toggleAxes = "toggleAxes";
		imap.put(space, toggleAxes);
		String increaseTimeScale = "+time";
		imap.put(rKey, increaseTimeScale);
		String decreaseTimeScale = "-time";
		imap.put(fKey, decreaseTimeScale);
		
		// get the action map for the center panel
		ActionMap amap = centerPanel.getActionMap();
		// put the "myCommand" command object into the central panel's action map
		amap.put(forwardKey, new ViewTranslate(canvas, 0, 0, 1));
		amap.put(backwardKey, new ViewTranslate(canvas, 0, 0, -1));
		amap.put(leftKey, new ViewTranslate(canvas, 1, 0, 0));
		amap.put(rightKey, new ViewTranslate(canvas, -1, 0, 0));
		amap.put(upKey, new ViewTranslate(canvas, 0, -1, 0));
		amap.put(downKey, new ViewTranslate(canvas, 0, 1, 0));
		amap.put(rotateDown, new ViewRotate(canvas, -1, 0, 0));
		amap.put(rotateUp, new ViewRotate(canvas, 1, 0, 0));
		amap.put(rotateLeft, new ViewRotate(canvas, 0, 1, 0));
		amap.put(rotateRight, new ViewRotate(canvas, 0, -1, 0));
		amap.put(rollRight, new ViewRotate(canvas, 0, 0, 1));
		amap.put(rollLeft, new ViewRotate(canvas, 0, 0, -1));
		amap.put(toggleAxes, new ViewToggleAxes(canvas));
		amap.put(increaseTimeScale, new ViewStepTimeScale(canvas, 10));
		amap.put(decreaseTimeScale, new ViewStepTimeScale(canvas, .1));
		// have the JFrame request keyboard focus
		this.requestFocus();

		// Show the finished window
		setVisible(true);

		// Instantiate the FPS Animator
		animator = new FPSAnimator(canvas, 144);
		// Start the FPS Animator. According to the documentation
		// this function usually blocks, so it needs to be done last.
		animator.start();
	}
}
