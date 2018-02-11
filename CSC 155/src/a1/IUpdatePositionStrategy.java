package a1;

/**
 * Strategy interface that provides the signature for updating the model's
 * position
 * 
 * @author Graeme Knowles
 */
@FunctionalInterface
public interface IUpdatePositionStrategy {
	 
	/**
	 * @param currentPosition The position of the model to be updated. The updating
	 * should be done in place.
	 */
	void updatePosition(float[] currentPosition);
}
