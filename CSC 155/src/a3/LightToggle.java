package a3;

import java.awt.event.ActionEvent;

public class LightToggle extends ViewAction {

	public LightToggle(View view) {
		super(view);
	}

	@Override
	public void actionPerformed(ActionEvent arg0) {
		view.toggleLight();	
	}
	
	
	
}
