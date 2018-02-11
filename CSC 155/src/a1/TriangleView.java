package a1;

import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.awt.GLCanvas;
import java.nio.*;
import static com.jogamp.opengl.GL.GL_VERSION;
import static com.jogamp.opengl.GL4.*;
import com.jogamp.common.nio.Buffers;

/**
 * The view of the model
 * Draws the triangle on the screen
 * 
 * @author Graeme Knowles
 */
public class TriangleView extends GLCanvas implements GLEventListener {

	private ModelProxy modelProxy = null;
	
	// Shader information
	private int vao[] = new int[1];
	private int renderingProgram = -1;

	/**
	 * The color the GLCanvas is cleared to
	 */
	private float backgroundColor[] = { 1, 1, 1, 1 };
	
	/**
	 * @param modelProxy The proxy to the model this view is observing
	 */
	public TriangleView(ModelProxy modelProxy) {
		this.modelProxy = modelProxy;
		addGLEventListener(this);
	}

	@Override
	public void display(GLAutoDrawable arg0) {
		GL4 gl = (GL4) GLContext.getCurrentGL();
		// Select the shader program to use
		gl.glUseProgram(renderingProgram);
		// Clear the background
		FloatBuffer bkgBuffer = Buffers.newDirectFloatBuffer(backgroundColor);
		gl.glClearBufferfv(GL_COLOR, 0, bkgBuffer);
		
		// Pass in the position
		int offsetLoc = gl.glGetUniformLocation(renderingProgram, "position");
		float[] position = modelProxy.getPosition();
		gl.glProgramUniform4f(renderingProgram, offsetLoc, position[0], position[1], position[2], position[3]);
		
		// Pass in the scale
		offsetLoc = gl.glGetUniformLocation(renderingProgram, "size");
		gl.glProgramUniform1f(renderingProgram, offsetLoc, modelProxy.getSize());

		// Pass in colors
		offsetLoc = gl.glGetUniformLocation(renderingProgram, "vertex1Color");
		float[] color1 = modelProxy.getColor1();
		gl.glProgramUniform4f(renderingProgram, offsetLoc, color1[0], color1[1], color1[2], color1[3]);		
		offsetLoc = gl.glGetUniformLocation(renderingProgram, "vertex2Color");
		float[] color2 = modelProxy.getColor2();
		gl.glProgramUniform4f(renderingProgram, offsetLoc, color2[0], color2[1], color2[2], color2[3]);
		offsetLoc = gl.glGetUniformLocation(renderingProgram, "vertex3Color");
		float[] color3 = modelProxy.getColor3();
		gl.glProgramUniform4f(renderingProgram, offsetLoc, color3[0], color3[1], color3[2], color3[3]);

		// Draw the triangle
		gl.glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	@Override
	public void dispose(GLAutoDrawable arg0) {
		// empty implementation
	}

	@Override
	public void init(GLAutoDrawable arg0) {
		GL4 gl = (GL4) GLContext.getCurrentGL();
		// Since this is the first location the GLContext is guaranteed to be initialized
		// Print out the JOGL and open GL version
		System.out.println("JOGL Version: " + Util.JOGL.getJOGLVersion());
		System.out.println("OpenGL Version: " + gl.glGetString(GL_VERSION));
		
		// Load and generate the shader program
		renderingProgram = Util.OGL.generateShaderProgram("src/a1/vertex.shader", "src/a1/fragment.shader");
		gl.glGenVertexArrays(vao.length, vao, 0);
		gl.glBindVertexArray(vao[0]);
	}

	@Override
	public void reshape(GLAutoDrawable arg0, int arg1, int arg2, int arg3, int arg4) {
		// empty implementation
	}
}
