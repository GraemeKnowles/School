package a3;

import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_LINES;
import static com.jogamp.opengl.GL.GL_STATIC_DRAW;

import java.nio.FloatBuffer;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;

import graphicslib3D.Vertex3D;

public class Primitives {

	public static class Line extends Model3D {
		public Line(float x, float y, float z, String texturePath) {
			super(new float[] { 0, 0, 0, x, y, z }, new float[] { 0, 0, 0, 1, 1, 1 }, texturePath);
			this.setVertexType(GL_LINES);
		}
	}

	public static class Tetrahedron extends Model3D {
		public Tetrahedron(String texturePath) {
			super(texturePath);
			float srqt8o9 = (float) Math.sqrt(8f / 9f);
			float srqt2o9 = (float) Math.sqrt(2f / 9f);
			float srqt2o3 = (float) Math.sqrt(2f / 9f);
			float negOneOthree = -1f / 3f;
			// float[] v1 = { srqt8o9, 0, negOneOthree };
			// float[] v2 = { -srqt2o9, srqt2o3, negOneOthree };
			// float[] v3 = { -srqt2o9, -srqt2o3, negOneOthree };
			// float[] v4 = { 0, 0, 1 };
			float[] vertices = { -srqt2o9, -srqt2o3, negOneOthree, srqt8o9, 0, negOneOthree, 0, 0, 1, srqt8o9, 0,
					negOneOthree, -srqt2o9, srqt2o3, negOneOthree, 0, 0, 1, -srqt2o9, srqt2o3, negOneOthree, -srqt2o9,
					-srqt2o3, negOneOthree, 0, 0, 1, -srqt2o9, srqt2o3, negOneOthree, srqt8o9, 0, negOneOthree,
					-srqt2o9, -srqt2o3, negOneOthree };

			float[] textureCoords = { 0, 0, 1, 0, .5f, 1, 0, 0, 1, 0, .5f, 1, 0, 0, 1, 0, .5f, 1, 0, 0, 1, 0, .5f, 1 };

			this.setVertices(vertices);
			this.setTexCoords(textureCoords);
		}
	}

	public static class Pyramid extends Model3D {
		public Pyramid(String texturePath) {
			super(new float[] { -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
					// front
					1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // right
					1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // back
					-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, // left
					-1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, // LF
					1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f },
					new float[] { 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f,
							1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
							1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f },
					texturePath);
		}
	}

	public static class Sphere extends Model3D {

		private graphicslib3D.shape.Sphere source = new graphicslib3D.shape.Sphere();
		
		public Sphere(){
			setVertexArrays();
		}
		
		public Sphere(String texturePath) {
			super(texturePath);
			setVertexArrays();
		}
		
		private void setVertexArrays() {
			Vertex3D[] vertices = source.getVertices();
			int[] indices = source.getIndices();

			float[] pvalues = new float[indices.length * 3];
			float[] tvalues = new float[indices.length * 2];
			float[] nvalues = new float[indices.length * 3];

			for (int i = 0; i < indices.length; i++) {
				pvalues[i * 3] = (float) (vertices[indices[i]]).getX();
				pvalues[i * 3 + 1] = (float) (vertices[indices[i]]).getY();
				pvalues[i * 3 + 2] = (float) (vertices[indices[i]]).getZ();
				tvalues[i * 2] = (float) (vertices[indices[i]]).getS();
				tvalues[i * 2 + 1] = (float) (vertices[indices[i]]).getT();
				nvalues[i * 3] = (float) (vertices[indices[i]]).getNormalX();
				nvalues[i * 3 + 1] = (float) (vertices[indices[i]]).getNormalY();
				nvalues[i * 3 + 2] = (float) (vertices[indices[i]]).getNormalZ();
			}
			super.vertices = pvalues;
			super.texCoords = tvalues;
			super.normals = nvalues;
		}
	}
}
