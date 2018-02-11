package a3;

import java.awt.event.ActionEvent;

public class ViewToggleAxes extends ViewAction {

	public ViewToggleAxes(View view) {
		super(view);
	}

	@Override
	public void actionPerformed(ActionEvent arg0) {
		view.toggleAxisVisibility();
	}
}
