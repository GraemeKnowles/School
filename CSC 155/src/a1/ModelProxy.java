package a1;

/**
 * This class provides for read only access to a model for a view.
 * This preserves the inability of the view to modify the model
 * 
 * @author Graeme Knowles
 */
public class ModelProxy {

	/**
	 * The model being accessed
	 */
	private TriangleModel model = null;
	
	public ModelProxy(TriangleModel model) {
		this.model = model;
	}
	
	/**
	 * @return the first color pole
	 */
	public float[] getColor1() {
		return model.getColor1().clone();
	}
	
	/**
	 * @return the second color pole
	 */
	public float[] getColor2() {
		return model.getColor2().clone();
	}
	
	/**
	 * @return the third color pole
	 */
	public float[] getColor3() {
		return model.getColor3().clone();
	}
	
	/**
	 * @return the position of the triangle
	 */
	public float[] getPosition() {
		return model.updatePosition().clone();
	}
	
	/**
	 * @return the size of the triangle
	 */
	public float getSize() {
		return model.getSize();
	}
}
