package a2;

import graphicslib3D.Matrix3D;
import graphicslib3D.Vector3D;

public abstract class Camera {
	
	private Vector3D position = new Vector3D(0, 0, 0);
	private Vector3D uAxis = new Vector3D(1, 0, 0);
	private Vector3D vAxis = new Vector3D(0, 1, 0);
	private Vector3D nAxis = new Vector3D(0, 0, 1);

	private Matrix3D translationMatrix = new Matrix3D();
	private Matrix3D rotationMatrix = new Matrix3D();
	private Matrix3D viewMatrix = new Matrix3D();
	
	private float aspectRatio = 1;
	private float fov = 60;
	private float minClip = 0.1f;
	private float maxClip = 100000.0f;

	public void translate(double u, double v, double n) {
		double movementSpeed = Math.max(1, position.magnitude() / 10);
		
		Vector3D uTrans = uAxis.mult(u * movementSpeed);
		Vector3D vTrans = vAxis.mult(v * movementSpeed);
		Vector3D nTrans = nAxis.mult(n * movementSpeed);
		position = position.add(uTrans).add(vTrans).add(nTrans);
		translationMatrix.setToIdentity();
		translationMatrix.translate(position.getX(), position.getY(), position.getZ());
		updateCombinedMatrix();
	}

	public void rotate(double u, double v, double n) {
		Matrix3D rotationStep = new Matrix3D();
		rotationStep.rotate(u, uAxis);
		rotationStep.rotate(v, vAxis);
		rotationStep.rotate(n, nAxis);
		uAxis = uAxis.mult(rotationStep);
		vAxis = vAxis.mult(rotationStep);
		nAxis = nAxis.mult(rotationStep);
		rotationMatrix.concatenate(rotationStep.inverse());
		updateCombinedMatrix();
	}

	protected void updateCombinedMatrix() {
		viewMatrix.setToIdentity();
		viewMatrix.concatenate(rotationMatrix);
		viewMatrix.concatenate(translationMatrix);
	}

	public Matrix3D getCombinedMatrix() {
		return viewMatrix;
	}
	
	public Matrix3D getTranslationMatrix() {
		return (Matrix3D) translationMatrix.clone();
	}

	public Matrix3D getRotationMatrix() {
		return (Matrix3D) rotationMatrix.clone();
	}
	
	public void resetRotation() {
		uAxis = new Vector3D(1, 0, 0);
		vAxis = new Vector3D(0, 1, 0);
		nAxis = new Vector3D(0, 0, 1);
		rotationMatrix.setToIdentity();
		updateCombinedMatrix();
	}
	
	public void resetTranslation() {
		position = new Vector3D(0, 0, 0);
		translationMatrix.setToIdentity();
		updateCombinedMatrix();
	}
	
	public float getAspectRatio() {
		return aspectRatio;
	}

	public void setAspectRatio(float aspectRatio) {
		this.aspectRatio = aspectRatio;
		updateCombinedMatrix();
	}
	
	public float getFov() {
		return fov;
	}

	public void setFov(float fov) {
		this.fov = fov;
		updateCombinedMatrix();
	}

	public float getMinClip() {
		return minClip;
	}

	public void setMinClip(float minClip) {
		this.minClip = minClip;
		updateCombinedMatrix();
	}

	public float getMaxClip() {
		return maxClip;
	}

	public void setMaxClip(float maxClip) {
		this.maxClip = maxClip;
		updateCombinedMatrix();
	}
}
