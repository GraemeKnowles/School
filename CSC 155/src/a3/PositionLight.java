package a3;

import a3.Primitives.Sphere;
import graphicslib3D.Material;
import graphicslib3D.Point3D;
import graphicslib3D.light.PositionalLight;

public class PositionLight extends PositionalLight
{
	private float[] amb;
	private float catt;
	private float[] diff;
	private float latt;
	private float qatt;
	private float[] spec;
	
	private ModelObject displayableObj = null;
	
	public PositionLight() {
		amb = new float[] {0,0,0,0};
		catt = getConstantAtt();
		diff = new float[] {0,0,0,0};
		latt = getLinearAtt();
		qatt = getQuadraticAtt();
		spec = new float[] {0,0,0,0};
		
		displayableObj  = new ModelObject(new Sphere(Util.Path.path + "/dYellow.jpg"), null);
		displayableObj.scale(0.01f, 0.01f, 0.01f);
		displayableObj.setBlocksLight(false);
		Material thisMat = new Material();
		thisMat.setAmbient(new float[] {1f,1f,1f,1f});
		thisMat.setDiffuse(new float[] {1f,1f,1f,1f});
		thisMat.setShininess(1);
		displayableObj.setMaterial(thisMat);
		displayableObj.setAffectedByLighting(false);
	}

	public void toggleLight() {
		float[] tamb = getAmbient();
		float tcatt = getConstantAtt();
		float[] tdiff = getDiffuse();
		float tlatt = getLinearAtt();
		float tqatt = getQuadraticAtt();
		float[] tspec = getSpecular();
		
		this.setAmbient(amb);
		this.setConstantAtt(catt);
		this.setDiffuse(diff);
		this.setLinearAtt(latt);
		this.setQuadraticAtt(qatt);
		this.setSpecular(spec);
		
		amb = tamb;
		catt = tcatt;
		diff = tdiff;
		latt = tlatt;
		qatt = tqatt;
		spec = tspec;
	}
	
	public void translate(double x, double y, double z) {
		Point3D position = getPosition();
		position.setX(position.getX() + x);
		position.setY(position.getY() + y);
		position.setZ(position.getZ() + z);
		setPosition(position);
		
		displayableObj.translate(x, y, z);
	}
	
	public ModelObject getDisplayableObj() {
		return displayableObj;
	}

	public void setDisplayableObj(ModelObject displayableObj) {
		this.displayableObj = displayableObj;
	}
}
