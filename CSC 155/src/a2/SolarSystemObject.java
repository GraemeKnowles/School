package a2;

import graphicslib3D.Matrix3D;
import graphicslib3D.Vector3D;

public class SolarSystemObject extends ModelObject {

	private Vector3D dailyRotMag = new Vector3D();

	private double orbitRadius = 0;
	private double orbitalSpeed = 0;
	private double orbitRotation = 0;

	public SolarSystemObject(float[] vertices, float[] textureVertices, String texturePath, ModelObject parent) {
	}

	public SolarSystemObject(Model3D model, ModelObject parent) {
		super(model, parent);
		dailyRotMag.setX(0);
		dailyRotMag.setY(0);
		dailyRotMag.setZ(0);
	}

	@Override
	public void init() {

	}

	@Override
	public void updateSelf(double deltaT) {
		// Orbit
		if (orbitRadius != 0) {
			orbitRotation = (orbitRotation + orbitalSpeed * deltaT) % 360;
			double x = orbitRadius * Math.cos(orbitRotation);
			double y = 0;
			double z = orbitRadius * Math.sin(orbitRotation);

			Matrix3D transMat = new Matrix3D();
			transMat.translate(x, y, z);
			this.setTranslation(transMat);
		}

		// Daily Rotation
		Vector3D rotStep = dailyRotMag.mult(deltaT);
		if (rotStep.magnitude() != 0) {
			Matrix3D rotMatrix = this.getRotation();
			rotMatrix.rotate(rotStep.magnitude(), rotStep);
			this.setRotation(rotMatrix);
		}
	}

	public Vector3D getDailyRotMag() {
		return dailyRotMag;
	}

	public void setDailyRotMag(Vector3D dailyRotMag) {
		this.dailyRotMag = dailyRotMag;
	}

	public double getOrbitRadius() {
		return orbitRadius;
	}

	public void setOrbitRadius(double orbitRadius) {
		this.orbitRadius = orbitRadius;
	}

	public double getOrbitalSpeed() {
		return orbitalSpeed;
	}

	public void setOrbitalSpeed(double orbitalSpeed) {
		this.orbitalSpeed = orbitalSpeed;
	}
}
