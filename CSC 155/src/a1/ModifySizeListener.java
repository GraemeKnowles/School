package a1;

import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

/**
 * This class listens for mouse wheel input and implements the following requirement from the Assignment 1 description:
 * 		mouse wheel control increases and decreases the size of the triangle
 * 
 * @author Graeme Knowles
 */
public class ModifySizeListener implements MouseWheelListener {
	
	/**
	 * The model being acted on
	 */
	private TriangleModel model = null;
	
	/**
	 * @param model the model to act on
	 */
	public ModifySizeListener(TriangleModel model) {
		this.model = model;
	}
	
	/* (non-Javadoc)
	 * @see java.awt.event.MouseWheelListener#mouseWheelMoved(java.awt.event.MouseWheelEvent)
	 * 
	 * This function determines what happens whenever the mouse wheel is moved
	 */
	@Override
	public void mouseWheelMoved(MouseWheelEvent e) {
		// Get the rotation, which is +-1 and modify it to a reasonable speed
		float delta = e.getWheelRotation() * .1f;
		// Set the scale of the model
		float modelScale = model.getSize() + delta;
		// Enforce the lower limit of the model's scale
		if(modelScale > 0)
		{
			model.setSize(modelScale);
		}
	}
}
