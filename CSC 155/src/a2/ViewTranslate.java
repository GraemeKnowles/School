package a2;

import java.awt.event.ActionEvent;

public class ViewTranslate extends ViewAction {

	double x, y, z;

	public ViewTranslate(View view, double x, double y, double z) {
		super(view);
		this.x = x;
		this.y = y;
		this.z = z;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		view.translateView(x, y, z);
	}
}
