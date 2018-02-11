package a3;

import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_STATIC_DRAW;
import static com.jogamp.opengl.GL.GL_TRIANGLES;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.util.texture.Texture;

import a3.Util.JOGL.ModelImporter;
import graphicslib3D.Vertex3D;

public class Model3D {
	protected float[] vertices = null;
	protected float[] texCoords = null;
	protected float[] normals = null;
	protected int vertexBufferID = -1;
	protected int texCoordID = -1;
	protected int normalID = -1;

	private Texture texture = null;
	private int vertexType = GL_TRIANGLES;

	protected Model3D() {
	}

	public Model3D(String modelPath, String texturePath) {
		
		ModelImporter mi = new ModelImporter();
		try {
			mi.parseOBJ(modelPath);
		} catch (IOException e) {
			System.out.println("Model not found: " + modelPath);
			System.exit(-1);
		}

		this.vertices = mi.getVertices();
		this.texCoords = mi.getTextureCoordinates();
		this.normals = mi.getNormals();

		if (texturePath != null) {
			texture = Util.JOGL.loadTexture(texturePath);
		}
	}

	public Model3D(ModelImporter mi, String texturePath) {
		this(mi.getVertices(), mi.getTextureCoordinates(), mi.getNormals(), texturePath);
	}

	public Model3D(float[] vertices, float[] texCoords, float[] normals, String texturePath) {
		this(texturePath);

		this.vertices = vertices;
		this.texCoords = texCoords;
		this.normals = normals;
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

		int[] vbo = new int[3];
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

		if (normals != null) {
			normalID = vbo[2];
			gl.glBindBuffer(GL_ARRAY_BUFFER, normalID);
			FloatBuffer norBuf = Buffers.newDirectFloatBuffer(normals);
			gl.glBufferData(GL_ARRAY_BUFFER, norBuf.limit() * 4, norBuf, GL_STATIC_DRAW);
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

	public float[] getNormals() {
		return normals;
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

	public int getNormalID() {
		return normalID;
	}
}
