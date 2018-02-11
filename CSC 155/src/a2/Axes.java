package a2;

import a2.Primitives.Line;

public class Axes extends ModelObject {
	@Override
	public void init() {
		final int AXIS_LENGTH = 100000;
		Line xAxisModel = new Line(AXIS_LENGTH, 0, 0, "src/red.jpg");
		ModelObject xAxis = new ModelObject(xAxisModel, this);
		Line yAxisModel = new Line(0, AXIS_LENGTH, 0, "src/green.jpg");
		ModelObject yAxis = new ModelObject(yAxisModel, this);
		Line zAxisModel = new Line(0, 0, AXIS_LENGTH, "src/blue.jpg");
		ModelObject zAxis = new ModelObject(zAxisModel, this);
	}
}
