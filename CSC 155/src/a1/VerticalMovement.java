package a1;

/**
 * This class is simultaneously an action and a strategy.
 * A member of this class, upon being activated sets the model's get position strategy to itself.
 * The strategy of this class is to move the position in a vertical manner
 * 
 * @author Graeme Knowles
 */
public class VerticalMovement extends ActionStrategy {
	
	/**
	 * A toggling boolean to determine the direction the position should move
	 */
	private boolean direction = false;
	
	/**
	 * @param model The model the action will act on, and who will use this strategy
	 */
	public VerticalMovement(TriangleModel model) {
		super("Vertical", model);
	}

	/* (non-Javadoc)
	 * @see a1.IUpdatePositionStrategy#updatePosition(float[])
	 * 
	 * Gets the next position of the model based on the current position
	 */
	@Override
	public void updatePosition(float[] currentPosition) {
		// Get the current position plus the direction delta
		float nextY = getNextY(currentPosition[1]);
		
		// Check if the next position is out of bounds
		if(nextY <= -1 || nextY >= 1) {
			// if it is, toggle the direction and get the new position
			direction = !direction;
			nextY = getNextY(currentPosition[1]);
		}
		
		// Modify the current position
		currentPosition[1] = nextY;
	}
	
	/**
	 * @param currentY the position to move from
	 * @return the next position
	 */
	private float getNextY(float currentY) {
		return currentY + (direction ? .01f : -.01f);
	}

}
