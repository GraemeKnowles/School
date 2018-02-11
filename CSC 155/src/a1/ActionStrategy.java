package a1;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

/**
 * This class defines an AbstractAction that acts upon a Model
 * 
 * @author Graeme Knowles
 */
public abstract class ActionStrategy extends ActionOnModel implements IUpdatePositionStrategy {

	@Override
	public void actionPerformed(ActionEvent arg0) {
		model.setGetPositionStrategy(this);
	}
	
	/**
	 * @param actionName The name of the action passed to AbstractAction
	 * @param model the model to act on
	 */
	public ActionStrategy(String actionName, TriangleModel model) {
		super(actionName, model);
	}
}
