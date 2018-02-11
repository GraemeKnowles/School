package a2;

import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_STATIC_DRAW;
import static com.jogamp.opengl.GL.GL_TRIANGLES;

import java.nio.FloatBuffer;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.util.texture.Texture;

public class Model3D {
	protected float[] vertices = null;
	protected float[] texCoords = null;
	protected int vertexBufferID = -1;
	protected int texCoordID = -1;
	private Texture texture = null;
	private int vertexType = GL_TRIANGLES;

	protected Model3D() {
	}

	public Model3D(String texturePath) {
		if (texturePath != null) {
			texture = Util.JOGL.loadTexture(texturePath);
		}
	}

	public Model3D(float[] vertices) {
		this.vertices = vertices;
	}

	public Model3D(float[] vertices, float[] texCoords, String texturePath) {
		this(texturePath);
		this.vertices = vertices;
		this.texCoords = texCoords;
	}

	public void initGL(GLAutoDrawable arg0) {
		GL4 gl = (GL4) GLContext.getCurrentGL();
		int[] vbo = new int[2];
		gl.glGenBuffers(vbo.length, vbo, 0);

		if (vertices != null) {
			vertexBufferID = vbo[0];
			gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
			FloatBuffer vertexBuffer = Buffers.newDirectFloatBuffer(vertices);
			gl.glBufferData(GL_ARRAY_BUFFER, vertexBuffer.limit() * 4, vertexBuffer, GL_STATIC_DRAW);
		}

		if (texCoords != null) {
			texCoordID = vbo[1];
			gl.glBindBuffer(GL_ARRAY_BUFFER, texCoordID);
			FloatBuffer texBuf = Buffers.newDirectFloatBuffer(texCoords);
			gl.glBufferData(GL_ARRAY_BUFFER, texBuf.limit() * 4, texBuf, GL_STATIC_DRAW);
		}
	}

	public int getVertexBufferID() {
		return vertexBufferID;
	}

	public int getTexCoordID() {
		return texCoordID;
	}

	public Texture getTexture() {
		return texture;
	}

	public void setTexture(Texture texture) {
		this.texture = texture;
	}

	public int getVertexCount() {
		return vertices.length / 3;
	}

	public int getVertexType() {
		return vertexType;
	}

	public void setVertexType(int vertexType) {
		this.vertexType = vertexType;
	}
	
	protected float[] getVertices() {
		return vertices;
	}

	protected void setVertices(float[] vertices) {
		this.vertices = vertices;
	}

	protected float[] getTexCoords() {
		return texCoords;
	}

	protected void setTexCoords(float[] texCoords) {
		this.texCoords = texCoords;
	}
}
