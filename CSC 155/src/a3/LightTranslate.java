package a3;

import java.awt.event.ActionEvent;

public class LightTranslate extends ViewAction {

	static final double sensitivity = 0.1;
	double x, y, z;

	public LightTranslate(View view, double x, double y, double z) {
		super(view);
		this.x = x;
		this.y = y;
		this.z = z;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		view.translateLight(x * sensitivity, y * sensitivity, z * sensitivity);
	}
}
