package a3;

import static com.jogamp.opengl.GL2ES3.GL_COLOR;
import static com.jogamp.opengl.GL.*;
import static com.jogamp.opengl.GL4.*;

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
import graphicslib3D.Material;
import graphicslib3D.Matrix3D;
import graphicslib3D.MatrixStack;
import graphicslib3D.Point3D;
import graphicslib3D.Vector3D;
import graphicslib3D.light.AmbientLight;
import graphicslib3D.light.Light;
import graphicslib3D.light.PositionalLight;

import graphicslib3D.*;
import graphicslib3D.light.*;
import graphicslib3D.GLSLUtils.*;
import graphicslib3D.shape.*;

import java.nio.*;
import javax.swing.*;

import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*;

/**
 * The view of the model Draws the triangle on the screen
 *
 * @author Graeme Knowles
 */
public class View extends GLCanvas implements GLEventListener {
	private Axes worldAxes = new Axes();
	private List<ModelObject> objects = new LinkedList<ModelObject>();

	private Scene scene;

	// Shader information
	private int mainRenderingProgram = -1;
	private int shadowRenderingProgram = -1;

	private PerspectiveCamera camera = null;

	private double lastTime = -1;
	private double timeScale = 0.001;

	private int shadow_tex = -1;
	private int shadow_buffer = -1;
	private Matrix3D b = new Matrix3D();

	/**
	 * The color the GLCanvas is cleared to
	 */
	private float backgroundColor[] = { 0, 0, 0, 1 };

	/**
	 * @param model
	 *            The main model this view is observing
	 */
	public View(Scene model, PerspectiveCamera camera) {
		objects.add(worldAxes);
		objects.add(model);
		scene = model;
		this.camera = camera;
		addGLEventListener(this);
	}

	@Override
	public void display(GLAutoDrawable arg0) {
		GL4 gl = (GL4) GLContext.getCurrentGL();
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

		gl.glBindFramebuffer(GL_FRAMEBUFFER, shadow_buffer);
		gl.glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_tex, 0);

		gl.glDrawBuffer(GL_NONE);
		gl.glEnable(GL_DEPTH_TEST);

		gl.glEnable(GL_POLYGON_OFFSET_FILL); // for reducing
		gl.glPolygonOffset(2.0f, 4.0f); // shadow artifacts

		gl.glUseProgram(shadowRenderingProgram);
		// Shadow Pass
		for (ModelObject obj : objects) {
			drawObject(gl, shadowRenderingProgram, obj, true);
		}

		gl.glDisable(GL_POLYGON_OFFSET_FILL); // artifact reduction, continued

		gl.glBindFramebuffer(GL_FRAMEBUFFER, 0);
		gl.glActiveTexture(GL_TEXTURE1);
		gl.glBindTexture(GL_TEXTURE_2D, shadow_tex);

		gl.glDrawBuffer(GL_FRONT);

		gl.glUseProgram(mainRenderingProgram);
		// Draw Pass
		gl.glClear(GL_DEPTH_BUFFER_BIT);
		// Draw light obj
		drawObject(gl, mainRenderingProgram, scene.getLight().getDisplayableObj(), false);
		for (ModelObject obj : objects) {
			drawObject(gl, mainRenderingProgram, obj, false);
		}
	}

	private void drawObject(GL4 gl, int renderProg, ModelObject obj, final boolean shadowPass) {
		// Perspective/View Matrix
		MatrixStack stack = new MatrixStack(obj.getObjectCount() * 6);
		stack.pushMatrix();
		stack.multMatrix(camera.getViewMatrix());

		// Light/Shadow matrix
		MatrixStack shadowStack = new MatrixStack(obj.getObjectCount() * 6);
		final Point3D origin = new Point3D(0.0, 0.0, 0.0);
		final Vector3D up = new Vector3D(0.0, 1.0, 0.0);
		shadowStack.pushMatrix();
		shadowStack.multMatrix(camera.getPerspectiveMatrix());
		shadowStack.pushMatrix();
		shadowStack.multMatrix(Util.JOGL.lookAt(scene.getLight().getPosition(), origin, up));
		if (shadowPass) {
			gl.glClear(GL_DEPTH_BUFFER_BIT);
		}
		drawObject(gl, renderProg, obj, stack, shadowStack, shadowPass);
	}

	private void drawObject(GL4 gl, int renderingProgram, ModelObject obj, MatrixStack stack, MatrixStack shadowStack,
			final boolean shadowPass) {
		if (!obj.isEnabled()) {
			return;
		}

		// push translation
		stack.pushMatrix();
		stack.multMatrix(obj.getTranslation());
		shadowStack.pushMatrix();
		shadowStack.multMatrix(obj.getTranslation());

		if (obj.drawable()) {
			// push rotation
			stack.pushMatrix();
			stack.multMatrix(obj.getRotation());
			shadowStack.pushMatrix();
			shadowStack.multMatrix(obj.getRotation());

			// push scale
			stack.pushMatrix();
			stack.multMatrix(obj.getScale());
			shadowStack.pushMatrix();
			shadowStack.multMatrix(obj.getScale());

			Matrix3D shadowMVP = shadowStack.peek();

			if (!shadowPass) {
				Matrix3D temp = (Matrix3D) b.clone();
				temp.concatenate(shadowMVP);
				shadowMVP = temp;

				// pass matrix
				final int mvLoc = gl.glGetUniformLocation(renderingProgram, "mv_matrix");
				gl.glUniformMatrix4fv(mvLoc, 1, false, stack.peek().getFloatValues(), 0);

				float[] normals = obj.getNormals();
				if (normals != null) {
					final int nLoc = gl.glGetUniformLocation(renderingProgram, "norm_matrix");
					gl.glUniformMatrix4fv(nLoc, 1, false, (stack.peek().inverse()).transpose().getFloatValues(), 0);
				}

				final int proj_location = gl.glGetUniformLocation(renderingProgram, "proj_matrix");
				gl.glUniformMatrix4fv(proj_location, 1, false, camera.getPerspectiveMatrix().getFloatValues(), 0);

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

				if (obj.getNormals() != null) {
					gl.glBindBuffer(GL_ARRAY_BUFFER, obj.getNormalID());
					gl.glVertexAttribPointer(2, 3, GL_FLOAT, false, 0, 0);
					gl.glEnableVertexAttribArray(2);
				}

				if (obj.isAffectedByLighting()) {
					// set the current globalAmbient settings
					final int globalAmbLoc = gl.glGetUniformLocation(renderingProgram, "globalAmbient");
					gl.glProgramUniform4fv(renderingProgram, globalAmbLoc, 1,
							AmbientLight.getAmbientLight().getValues(), 0);

					installLight(renderingProgram, camera.getViewMatrix(), scene.getLight(), obj.getMaterial());
				}
			}

			final int shadow_location = gl.glGetUniformLocation(renderingProgram, "shadowMVP");
			gl.glUniformMatrix4fv(shadow_location, 1, false, shadowMVP.getFloatValues(), 0);

			gl.glEnable(GL_CULL_FACE);
			gl.glFrontFace(GL_CCW);
			gl.glEnable(GL_DEPTH_TEST);
			gl.glDepthFunc(GL_LEQUAL);

			// Draw the arrays
			gl.glDrawArrays(obj.getVertexType(), 0, obj.getVertexCount());

			// pop scale
			stack.popMatrix();
			shadowStack.popMatrix();
			// pop rotation
			stack.popMatrix();
			shadowStack.popMatrix();
			// do not pop translation to allow it to affect sub-objects
		}

		// draw children
		for (ModelObject child : obj.getChildren()) {
			drawObject(gl, renderingProgram, child, stack, shadowStack, shadowPass);
		}

		// Cleanup the stack
		// pop translation
		stack.popMatrix();
		shadowStack.popMatrix();
	}

	@Override
	public void dispose(GLAutoDrawable arg0) {
		// empty implementation
	}

	@Override
	public void init(GLAutoDrawable arg0) {
		lastTime = System.currentTimeMillis();

		// Load and generate the shader program
		mainRenderingProgram = Util.JOGL.generateShaderProgram(Util.Path.pathPack + "/vertex.shader",
				Util.Path.pathPack + "/fragment.shader");
		shadowRenderingProgram = Util.JOGL.generateShaderProgram(Util.Path.pathPack + "/vertexShadow.shader", null);

		for (ModelObject obj : objects) {
			obj.initGL(arg0);
		}

		Point3D camPos = scene.getPreferredCameraPosition();
		this.camera.translate(camPos.getX(), camPos.getY(), camPos.getZ());
		Point3D camRot = scene.getPreferredCameraRotation();
		this.camera.rotate(camRot.getX(), camRot.getY(), camRot.getZ());

		setupShadowBuffers();

		b.setElementAt(0, 0, 0.5);
		b.setElementAt(0, 1, 0.0);
		b.setElementAt(0, 2, 0.0);
		b.setElementAt(0, 3, 0.5f);
		b.setElementAt(1, 0, 0.0);
		b.setElementAt(1, 1, 0.5);
		b.setElementAt(1, 2, 0.0);
		b.setElementAt(1, 3, 0.5f);
		b.setElementAt(2, 0, 0.0);
		b.setElementAt(2, 1, 0.0);
		b.setElementAt(2, 2, 0.5);
		b.setElementAt(2, 3, 0.5f);
		b.setElementAt(3, 0, 0.0);
		b.setElementAt(3, 1, 0.0);
		b.setElementAt(3, 2, 0.0);
		b.setElementAt(3, 3, 1.0f);

		// may reduce shadow border artifacts
		GL4 gl = (GL4) GLContext.getCurrentGL();
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		
		scene.getLight().getDisplayableObj().initGL(arg0);
	}

	@Override
	public void reshape(GLAutoDrawable arg0, int arg1, int arg2, int arg3, int arg4) {
		camera.setAspectRatio((float) getWidth() / (float) getHeight());
		setupShadowBuffers();
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

	public void toggleLight() {
		scene.getLight().toggleLight();
	}

	public void translateLight(double x, double y, double z) {
		scene.getLight().translate(x, y, z);
	}

	private void installLight(int renderProg, Matrix3D v_matrix, PositionalLight light, Material material) {
		GL4 gl = (GL4) GLContext.getCurrentGL();

		Point3D lightP = light.getPosition();
		Point3D lightPv = lightP.mult(v_matrix);
		float[] viewspaceLightPos = new float[] { (float) lightPv.getX(), (float) lightPv.getY(),
				(float) lightPv.getZ() };

		// get the locations of the light and material fields in the shader
		final int ambLoc = gl.glGetUniformLocation(renderProg, "light.ambient");
		final int diffLoc = gl.glGetUniformLocation(renderProg, "light.diffuse");
		final int specLoc = gl.glGetUniformLocation(renderProg, "light.specular");
		final int posLoc = gl.glGetUniformLocation(renderProg, "light.position");
		final int MambLoc = gl.glGetUniformLocation(renderProg, "material.ambient");
		final int MdiffLoc = gl.glGetUniformLocation(renderProg, "material.diffuse");
		final int MspecLoc = gl.glGetUniformLocation(renderProg, "material.specular");
		final int MshiLoc = gl.glGetUniformLocation(renderProg, "material.shininess");

		// set the uniform light and material values in the shader
		gl.glProgramUniform4fv(renderProg, ambLoc, 1, light.getAmbient(), 0);
		gl.glProgramUniform4fv(renderProg, diffLoc, 1, light.getDiffuse(), 0);
		gl.glProgramUniform4fv(renderProg, specLoc, 1, light.getSpecular(), 0);
		gl.glProgramUniform3fv(renderProg, posLoc, 1, viewspaceLightPos, 0);
		gl.glProgramUniform4fv(renderProg, MambLoc, 1, material.getAmbient(), 0);
		gl.glProgramUniform4fv(renderProg, MdiffLoc, 1, material.getDiffuse(), 0);
		gl.glProgramUniform4fv(renderProg, MspecLoc, 1, material.getSpecular(), 0);
		gl.glProgramUniform1f(renderProg, MshiLoc, material.getShininess());
	}

	public void setupShadowBuffers() {
		GL4 gl = (GL4) GLContext.getCurrentGL();

		if (shadow_buffer == -1) {
			int[] buff = new int[1];
			gl.glGenFramebuffers(1, buff, 0);
			shadow_buffer = buff[0];
		}

		if (shadow_tex == -1) {
			int[] tex = new int[1];
			gl.glGenTextures(1, tex, 0);
			shadow_tex = tex[0];
		}

		gl.glBindTexture(GL_TEXTURE_2D, shadow_tex);
		gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, getWidth(), getHeight(), 0, GL_DEPTH_COMPONENT,
				GL_FLOAT, null);
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
		gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	}
}
