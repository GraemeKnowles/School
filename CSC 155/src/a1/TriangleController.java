package a1;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JButton;

import java.awt.BorderLayout;
import java.awt.event.MouseWheelListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.*;
import com.jogamp.opengl.util.FPSAnimator;

/**
 * This class is the controller portion of the Model-View-Controller architecture
 * It acts upon a model (or creates objects that act upon it)
 * 
 * @author Graeme Knowles
 */
public class TriangleController extends JFrame{

	/**
	 * The model that contains all the information being displayed
	 */
	private TriangleModel model = null;
	
	// GUI

	/**
	 * The center panel of the window. This contains the OpenGL canvas
	 */
	private JPanel centerPanel = null;
	
	/**
	 * The View portion of the MVC architecture. This is the openGL canvas where the 
	 * Objects are drawn
	 */
	private TriangleView canvas = null;
	
	/**
	 * The bottom panel that contains the interactive buttons
	 */
	private JPanel botPanel = null;
	
	/**
	 * The label on the bottom panel
	 */
	private JLabel botPanelLabel = null;
	
	/**
	 * This list contains all of the buttons on the bottom panel in order
	 */
	private List<JButton> bottomPanelButtons = new ArrayList<JButton>();
	
	/**
	 * The action listener that listens for mouse wheel input
	 */
	private MouseWheelListener mouseListener = null;
	
	/**
	 * The object the drives the animation
	 */
	FPSAnimator animator = null;
	
	/**
	 * @param model The model the controller acts on
	 * @param view The view that displays the information from the model
	 */
	public TriangleController(TriangleModel model, TriangleView view) {
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
		setSize(800,800);
		setLocation(200,200);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		// Create + Add the objects
		BorderLayout centerPanelLayout = new BorderLayout();
		centerPanel = new JPanel(centerPanelLayout);
		// add the drawing canvas
		centerPanel.add(canvas);
		this.add(centerPanel, BorderLayout.CENTER);
		
		// the bottom panel
		botPanel = new JPanel();
		this.add(botPanel,BorderLayout.SOUTH);
		// bottom panel label
		botPanelLabel = new JLabel("Movement Type: ");
		botPanel.add(botPanelLabel);
		// a button that causes the triangle to move in a circle around the GLCanvas.
		JButton firstButton = new JButton ();
		CircularMovement acm = new CircularMovement(model);
		firstButton.setAction(acm);
		bottomPanelButtons.add(firstButton);
		// a button that causes the triangle to move up and down, vertically.
		JButton secondButton = new JButton ();
		VerticalMovement avm = new VerticalMovement(model);
		secondButton.setAction(avm);
		bottomPanelButtons.add(secondButton);
		// A stop button
		JButton thirdButton = new JButton ();
		NoMovement asnm = new NoMovement(model);
		thirdButton.setAction(asnm);
		bottomPanelButtons.add(thirdButton);
		
		for (JButton button : bottomPanelButtons) {
			this.botPanel.add(button);
		}

		// a key binding that toggles the triangle between a single solid color, 
		// and a gradient of three colors when user presses the ‘c’ key.
		// get the "focus is in the window" input map for the center panel
		int mapName = JComponent.WHEN_IN_FOCUSED_WINDOW;
		InputMap imap = centerPanel.getInputMap(mapName);
		// create a keystroke object to represent the "c" key
		KeyStroke cKey = KeyStroke.getKeyStroke('c');
		// put the "cKey" keystroke object into the central panel’s "when focus is
		// in the window" input map under the identifier name "color“
		String colorKey = "color";
		imap.put(cKey, colorKey);
		// get the action map for the center panel
		ActionMap amap = centerPanel.getActionMap();
		// put the "myCommand" command object into the central panel's action map
		ActionToggleColor atc = new ActionToggleColor(model);
		amap.put(colorKey, atc);
		//have the JFrame request keyboard focus
		this.requestFocus();
		
		// Set the mouse wheel to control the size of the model
		mouseListener = new ModifySizeListener(model);
		this.addMouseWheelListener(mouseListener);
		
		// Show the finished window
		setVisible(true);
		
		// Instantiate the FPS Animator
		animator = new FPSAnimator(canvas, 144);
		// Start the FPS Animator. According to the documentation
		// this function usually blocks, so it needs to be done last.
		animator.start();
	}
}
