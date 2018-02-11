package a3;


public class Scene extends ModelObject {
	private PositionLight light = null;

	public void setLight(PositionLight light) {
		this.light = light;
	}
	
	public PositionLight getLight() {
		return this.light;
	}
}
