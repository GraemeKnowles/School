package a2;

import javax.swing.AbstractAction;

public abstract class ViewAction extends AbstractAction {

	protected View view = null;

	public ViewAction(View view) {
		this.view = view;
	}
}
