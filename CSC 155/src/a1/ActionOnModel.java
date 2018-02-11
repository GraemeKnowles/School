package a1;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

public abstract class ActionOnModel extends AbstractAction {

	/**
	 * The Model the AbstractAction acts on
	 */
	protected TriangleModel model = null;
	
	/**
	 * @param actionName The name of the action passed to AbstractAction
	 * @param model the model to act on
	 */
	public ActionOnModel(String actionName, TriangleModel model) {
		super(actionName);
		this.model = model;
	}
}
