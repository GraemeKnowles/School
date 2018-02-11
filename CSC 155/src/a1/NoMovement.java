package a1;

public class NoMovement extends ActionStrategy {

	public NoMovement(TriangleModel model) {
		super("None", model);
	}

	@Override
	public void updatePosition(float[] currentPosition) {
		return;
	}
}
