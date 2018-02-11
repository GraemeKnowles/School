package a3;

import graphicslib3D.Matrix3D;

public class PerspectiveCamera extends Camera {
	private Matrix3D perspectiveMatrix = null;

	@Override
	protected void updateCombinedMatrix() {
		super.updateCombinedMatrix();
		perspectiveMatrix = generatePerspectiveMatrix(getFov(), getAspectRatio(), getMinClip(), getMaxClip());
	}

	private Matrix3D generatePerspectiveMatrix(float fovy, float aspect, float n, float f) {
		float q = 1.0f / ((float) Math.tan(Math.toRadians(0.5f * fovy)));
		float A = q / aspect;
		float B = (n + f) / (n - f);
		float C = (2.0f * n * f) / (n - f);
		Matrix3D r = new Matrix3D();
		r.setElementAt(0, 0, A);
		r.setElementAt(1, 1, q);
		r.setElementAt(2, 2, B);
		r.setElementAt(3, 2, -1.0f);
		r.setElementAt(2, 3, C);
		r.setElementAt(3, 3, 0.0f);
		return r;
	}

	public Matrix3D getPerspectiveMatrix() {
		return (Matrix3D) perspectiveMatrix.clone();
	}
}
