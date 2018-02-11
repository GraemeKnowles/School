package a1;

/**
 * This class is simultaneously an action and a strategy.
 * A member of this class, upon being activated sets the model's get position strategy to itself.
 * The strategy of this class is to move the position in a circular manner about the origin
 * 
 * @author Graeme Knowles
 */
public class CircularMovement extends ActionStrategy {

	/**
	 * The amount of radians 1 degree is equal to
	 */
	private static final double RAD2DEG = Math.toRadians(1);
	
	/**
	 * @param model The model this action will act upon, and who will use this strategy
	 */
	public CircularMovement(TriangleModel model) {
		super("Circular", model);
	}

	/* (non-Javadoc)
	 * @see a1.IUpdatePositionStrategy#updatePosition(float[])
	 * 
	 * Returns the next position based on the current position if it's moving in a circular path
	 */
	@Override
	public void updatePosition(float[] currentPosition) {
		float[] position = model.getCurrentPosition();
		// Convert to polar and increment the angle
		double angleRad = Math.atan2(position[1], position[0]) - RAD2DEG;
		double distanceFromCenter = Math.sqrt(position[0] * position[0] + position[1] * position[1]);
		// Convert back to Cartesian
		currentPosition[0] = (float) (distanceFromCenter * Math.cos(angleRad));
		currentPosition[1] = (float) (distanceFromCenter * Math.sin(angleRad));
	}

}
