package a1;

/**
 * The Model portion of the MVC architecture
 * This contains all of the information being acted on
 * 
 * @author Graeme Knowles
 */
public class TriangleModel {	
	
	/**
	 * The first color pole of the triangle's color gradient
	 */
	private float[] color1 = new float[] {1, 0, 0, 1};
	/**
	 * The second color pole of the triangle's color gradient
	 */
	private float[] color2 = new float[] {0, 1, 0, 1};
	/**
	 * The third color pole of the triangle's color gradient
	 */
	private float[] color3 = new float[] {0, 0, 1, 0};
	
	/**
	 * The position of the triangle in world space [-1,1]
	 */
	private float[] position = new float[] {.25f, .25f, .25f, 0};
	
	/**
	 * The size of the triangle as a percentage of the screen
	 */
	private float size = .25f;
	
	/**
	 * The strategy that determines how the triangle's position is updated
	 */
	private IUpdatePositionStrategy getPositionStratRef = null;
	
	/**
	 * @param ref The new strategy for the model to use when updating
	 */
	public void setGetPositionStrategy(IUpdatePositionStrategy ref) {
		getPositionStratRef = ref;
	}
	
	/**
	 * @return Gets the current position of the model
	 */
	public float[] getCurrentPosition() {
		return position;
	}
	
	/**
	 * Updates the position of the model
	 * @return the new position of the model
	 */
	public float[] updatePosition() {
		if(getPositionStratRef != null) {
			getPositionStratRef.updatePosition(position);
		}
		return position;
	}
	
	public float[] getColor1() {
		return color1;
	}
	
	public void setColor1(float[] newColor) {
		color1 = newColor;
	}
	
	public float[] getColor2() {
		return color2;
	}
	
	public void setColor2(float[] newColor) {
		color2 = newColor;
	}
	
	public float[] getColor3() {
		return color3;
	}
	
	public void setColor3(float[] newColor) {
		color3 = newColor;
	}

	public float getSize() {
		return size;
	}

	public void setSize(float size) {
		this.size = size;
	}
	
}
