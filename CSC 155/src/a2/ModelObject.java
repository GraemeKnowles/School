package a2;

import java.awt.Point;
import java.util.LinkedList;
import java.util.List;

import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.util.texture.Texture;

import graphicslib3D.Matrix3D;
import graphicslib3D.Point3D;
import graphicslib3D.Shape3D;

public class ModelObject extends Shape3D {
	private ModelObject parent = null;
	private List<ModelObject> children = new LinkedList<ModelObject>();

	private Model3D model = null;
	
	private boolean enabled = true;
	
	private Point3D preferredCameraPosition = new Point3D(0,0,0);

	/**
	 * Creates an empty ModelObject. This object is not displayable but can be used
	 * to create a hierarchy.
	 */
	public ModelObject() {

	}

	public ModelObject(Model3D model, ModelObject parent) {
		this.model = model;
		if (parent != null) {
			parent.addChild(this);
		}
	}

	public void initGL(GLAutoDrawable arg0) {
		this.init();

		if (model != null) {
			model.initGL(arg0);
		}

		for (ModelObject child : children) {
			child.initGL(arg0);
		}
	}

	public void addChild(ModelObject child) {
		children.add(child);
		child.parent = this;
	}

	public int getVertexID() {
		if (model == null) {
			return -1;
		}
		return model.getVertexBufferID();
	}

	public int getTextureCoordID() {
		if (model == null) {
			return -1;
		}
		return model.getTexCoordID();
	}

	public int getVertexCount() {
		if (model == null) {
			return 0;
		}

		return model.getVertexCount();
	}

	public Texture getTexture() {
		if (model == null) {
			return null;
		}
		return model.getTexture();
	}

	public void setScale(double scale) {
		Matrix3D scaleMat = new Matrix3D();
		scaleMat.scale(scale, scale, scale);
		setScale(scaleMat);
	}

	public int getObjectCount() {
		int count = 1;
		for (ModelObject child : children) {
			count += child.getObjectCount();
		}
		return count;
	}

	public List<ModelObject> getChildren() {
		return children;
	}

	public void update(double deltaTime) {
		updateSelf(deltaTime);
		for (ModelObject child : getChildren()) {
			child.update(deltaTime);
		}
	}

	public int getVertexType() {
		return model.getVertexType();
	}

	public void setVertexType(int vertexType) {
		model.setVertexType(vertexType);
	}

	public boolean drawable() {
		return this.getVertexID() >= 0;
	}

	public boolean texturable() {
		return this.getTextureCoordID() >= 0;
	}

	public void updateSelf(double deltaTime) {
	}

	public void init() {
	}

	public boolean isEnabled() {
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	public Point3D getPreferredCameraPosition() {
		return preferredCameraPosition;
	}

	public void setPreferredCameraPosition(Point3D preferredCameraPosition) {
		this.preferredCameraPosition = preferredCameraPosition;
	}
}
