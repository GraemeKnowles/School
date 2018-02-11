package a3;

import a3.Primitives.Line;
import graphicslib3D.Material;

public class Axes extends ModelObject {
	@Override
	public void init() {
		final int AXIS_LENGTH = 100000;
		Line xAxisModel = new Line(AXIS_LENGTH, 0, 0, Util.Path.path + "/red.jpg");
		ModelObject xAxis = new ModelObject(xAxisModel, this);
		Line yAxisModel = new Line(0, AXIS_LENGTH, 0, Util.Path.path + "/green.jpg");
		ModelObject yAxis = new ModelObject(yAxisModel, this);
		Line zAxisModel = new Line(0, 0, AXIS_LENGTH, Util.Path.path + "/blue.jpg");
		ModelObject zAxis = new ModelObject(zAxisModel, this);
		this.setAffectedByLighting(false);
		
		Material thisMat = new Material();
		thisMat.setAmbient(new float[] {1f,1f,1f,1f});
		thisMat.setDiffuse(new float[] {1f,1f,1f,1f});
		thisMat.setShininess(1);
		
		xAxis.setMaterial(thisMat);
		yAxis.setMaterial(thisMat);
		zAxis.setMaterial(thisMat);
		
		xAxis.setBlocksLight(false);
		yAxis.setBlocksLight(false);
		zAxis.setBlocksLight(false);
	}
}
