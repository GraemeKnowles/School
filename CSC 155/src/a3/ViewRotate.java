package a3;

import java.awt.event.ActionEvent;

public class ViewRotate extends ViewAction {

	double x, y, z;

	public ViewRotate(View view, double x, double y, double z) {
		super(view);
		this.x = x;
		this.y = y;
		this.z = z;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		view.rotateView(x, y, z);
	}
}
