package a2;

import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_CCW;
import static com.jogamp.opengl.GL.GL_CULL_FACE;
import static com.jogamp.opengl.GL.GL_DEPTH_BUFFER_BIT;
import static com.jogamp.opengl.GL.GL_DEPTH_TEST;
import static com.jogamp.opengl.GL.GL_FLOAT;
import static com.jogamp.opengl.GL.GL_TEXTURE0;
import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static com.jogamp.opengl.GL2ES3.GL_COLOR;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.List;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.awt.GLCanvas;

import a2.Primitives.Line;
import graphicslib3D.Matrix3D;
import graphicslib3D.MatrixStack;
import graphicslib3D.Point3D;

/**
 * The view of the model Draws the triangle on the screen
 *
 * @author Graeme Knowles
 */
public class View extends GLCanvas implements GLEventListener {
	private Axes worldAxes = new Axes();
	private List<ModelObject> objects = new LinkedList<ModelObject>();

	// Shader information
	private int renderingProgram = -1;

	private Camera camera = null;
	
	private double lastTime = -1;
	private double timeScale = 0.001;

	/**
	 * The color the GLCanvas is cleared to
	 */
	private float backgroundColor[] = { 0, 0, 0, 1 };

	/**
	 * @param model
	 *            The main model this view is observing
	 */
	public View(ModelObject model, Camera camera) {
		objects.add(worldAxes);
		objects.add(model);
		this.camera = camera;
		Point3D camPos = model.getPreferredCameraPosition();
		camera.translate(camPos.getX(), camPos.getY(), camPos.getZ());
		
		addGLEventListener(this);
	}

	@Override
	public void display(GLAutoDrawable arg0) {
		GL4 gl = (GL4) GLContext.getCurrentGL();
		// Select the shader program to use
		gl.glUseProgram(renderingProgram);
		// Clear the background
		gl.glClear(GL_DEPTH_BUFFER_BIT);
		FloatBuffer bkgBuffer = Buffers.newDirectFloatBuffer(backgroundColor);
		gl.glClearBufferfv(GL_COLOR, 0, bkgBuffer);
		gl.glClear(GL_DEPTH_BUFFER_BIT);

		double currentTime = System.currentTimeMillis();
		double deltaTime = (currentTime - lastTime) * timeScale;
		lastTime = currentTime;

		for (ModelObject obj : objects) {
			obj.update(deltaTime);
		}

		for (ModelObject obj : objects) {
			drawObject(gl, obj);
		}
	}

	private void drawObject(GL4 gl, ModelObject obj) {
		MatrixStack stack = new MatrixStack(obj.getObjectCount() * 6);
		// Perspective/View Matrix
		stack.pushMatrix();
		stack.multMatrix(camera.getCombinedMatrix());
		drawObject(gl, obj, stack);
	}

	private void drawObject(GL4 gl, ModelObject obj, MatrixStack stack) {
		if (!obj.isEnabled()) {
			return;
		}

		// push translation
		stack.pushMatrix();
		stack.multMatrix(obj.getTranslation());

		if (obj.drawable()) {
			// push rotation
			stack.pushMatrix();
			stack.multMatrix(obj.getRotation());

			// push scale
			stack.pushMatrix();
			stack.multMatrix(obj.getScale());

			// pass matrix
			int mvLoc = gl.glGetUniformLocation(renderingProgram, "mv_matrix");
			gl.glUniformMatrix4fv(mvLoc, 1, false, stack.peek().getFloatValues(), 0);

			// pop rotation
			stack.popMatrix();
			stack.popMatrix();
			// do not pop translation and scale to allow them to affect sub-objects

			// draw
			gl.glBindBuffer(GL_ARRAY_BUFFER, obj.getVertexID());
			gl.glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
			gl.glEnableVertexAttribArray(0);

			if (obj.texturable()) {
				// Texture Section
				gl.glBindBuffer(GL_ARRAY_BUFFER, obj.getTextureCoordID());
				gl.glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, 0);
				gl.glEnableVertexAttribArray(1);

				gl.glActiveTexture(GL_TEXTURE0);
				gl.glBindTexture(GL_TEXTURE_2D, obj.getTexture().getTextureObject());
			}

			gl.glEnable(GL_CULL_FACE);
			gl.glFrontFace(GL_CCW);

			// Draw the arrays
			gl.glEnable(GL_DEPTH_TEST);
			gl.glDrawArrays(obj.getVertexType(), 0, obj.getVertexCount());
		}

		// draw children
		for (ModelObject child : obj.getChildren()) {
			drawObject(gl, child, stack);
		}

		// Cleanup the stack
		// pop translation
		stack.popMatrix();
	}

	@Override
	public void dispose(GLAutoDrawable arg0) {
		// empty implementation
	}

	@Override
	public void init(GLAutoDrawable arg0) {
		lastTime = System.currentTimeMillis();

		// Load and generate the shader program
		renderingProgram = Util.JOGL.generateShaderProgram("src/a2/vertex.shader", "src/a2/fragment.shader");

		for (ModelObject obj : objects) {
			obj.initGL(arg0);
		}
	}

	@Override
	public void reshape(GLAutoDrawable arg0, int arg1, int arg2, int arg3, int arg4) {
		camera.setAspectRatio((float)getWidth() / (float) getHeight());
	}

	public void translateView(double x, double y, double z) {
		camera.translate(x, y, z);
	}

	public void rotateView(double x, double y, double z) {
		camera.rotate(x, y, z);
	}
	
	public void toggleAxisVisibility() {
		worldAxes.setEnabled(!worldAxes.isEnabled());
	}
	
	public double getTimeScale() {
		return timeScale;
	}

	public void setTimeScale(double timeScale) {
		this.timeScale = Math.max(0.000001, timeScale);
	}
}
