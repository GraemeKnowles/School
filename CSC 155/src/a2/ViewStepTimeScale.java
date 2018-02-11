package a2;

import java.awt.event.ActionEvent;

public class ViewStepTimeScale extends ViewAction {

	double stepAmount = 0;
	
	public ViewStepTimeScale(View view, double amount) {
		super(view);
		this.stepAmount = amount;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		view.setTimeScale(view.getTimeScale() * stepAmount);
	}

}
