package a1;

import java.awt.event.ActionEvent;

/**
 * An action that toggles the model's color
 * This class implements the following requirement from the Assignment 1 description:
 * 		A key binding that toggles the triangle between a single solid color and a gradient 
 * 		of three colors when user presses the ‘c’ key.
 * 
 * @author Graeme Knowles
 */
public class ActionToggleColor extends ActionOnModel {

	/**
	 * The single solid color to toggle to
	 */
	private float[] solidColor = new float[] {0, 0, 0, 1};
	
	/**
	 * The first color pole of the gradient to toggle to
	 */
	private float[] color1 = new float[] {1, 0, 0, 1};
	
	/**
	 * The second color pole of the gradient to toggle to
	 */
	private float[] color2 = new float[] {0, 1, 0, 1};
	
	/**
	 * The third color pole of the gradient to toggle to
	 */
	private float[] color3 = new float[] {0, 0, 1, 0};
	
	/**
	 * The boolean that is toggled back and forth to determine whether
	 * or not to toggle to the gradient or the solid color
	 */
	private boolean toggle = true;
	
	/**
	 * @param model The model this action will act on
	 */
	public ActionToggleColor(TriangleModel model) {
		super("Toggle Color", model);
	}
	
	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 * 
	 * Sets the model's colors to the specified color based on the toggle variable
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		toggle = !toggle;
		
		if(toggle) {
			model.setColor1(color1);
			model.setColor2(color2);
			model.setColor3(color3);
		}else {
			model.setColor1(solidColor);
			model.setColor2(solidColor);
			model.setColor3(solidColor);
		}
	}

}
