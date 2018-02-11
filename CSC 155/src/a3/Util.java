package a3;

import static com.jogamp.opengl.GL.GL_NO_ERROR;
import static com.jogamp.opengl.GL.GL_VERSION;
import static com.jogamp.opengl.GL2ES2.GL_FRAGMENT_SHADER;
import static com.jogamp.opengl.GL2ES2.GL_INFO_LOG_LENGTH;
import static com.jogamp.opengl.GL2ES2.GL_VERTEX_SHADER;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.glu.GLU;
import com.jogamp.opengl.util.texture.Texture;
import com.jogamp.opengl.util.texture.TextureIO;

import graphicslib3D.GLSLUtils;
import graphicslib3D.Matrix3D;
import graphicslib3D.Point3D;
import graphicslib3D.Vector3D;
import graphicslib3D.Vertex3D;

/**
 * Provides Utility/Helper functions for various aspects of the program
 *
 * @author Graeme Knowles
 */
class Util {

	static class Path {
		public static final String path = "src";
		public static final String packge = "a3";
		public static final String pathPack = path + "/" + packge;
	}

	/**
	 * @author Graeme Knowles
	 *
	 *         JOGL related functions
	 */
	static class JOGL {
		static String getJOGLVersion() {
			Package joglPack = Package.getPackage("com.jogamp.opengl");
			return joglPack.getImplementationVersion();
		}

		/**
		 * @return the OpenGL Version running
		 */
		static String getGLVersion() {
			GL4 gl = (GL4) GLContext.getCurrentGL();
			return gl.glGetString(GL_VERSION);
		}

		static Matrix3D lookAt(Point3D eye, Point3D target, Vector3D y) {
			Vector3D eyeV = new Vector3D(eye);
			Vector3D targetV = new Vector3D(target);
			Vector3D fwd = (targetV.minus(eyeV)).normalize();
			Vector3D side = (fwd.cross(y)).normalize();
			Vector3D up = (side.cross(fwd)).normalize();
			Matrix3D look = new Matrix3D();
			look.setElementAt(0, 0, side.getX());
			look.setElementAt(1, 0, up.getX());
			look.setElementAt(2, 0, -fwd.getX());
			look.setElementAt(3, 0, 0.0f);
			look.setElementAt(0, 1, side.getY());
			look.setElementAt(1, 1, up.getY());
			look.setElementAt(2, 1, -fwd.getY());
			look.setElementAt(3, 1, 0.0f);
			look.setElementAt(0, 2, side.getZ());
			look.setElementAt(1, 2, up.getZ());
			look.setElementAt(2, 2, -fwd.getZ());
			look.setElementAt(3, 2, 0.0f);
			look.setElementAt(0, 3, side.dot(eyeV.mult(-1)));
			look.setElementAt(1, 3, up.dot(eyeV.mult(-1)));
			look.setElementAt(2, 3, (fwd.mult(-1)).dot(eyeV.mult(-1)));
			look.setElementAt(3, 3, 1.0f);
			return (look);
		}

		/**
		 *
		 * @param vertexShader
		 *            the path of the vertex shader to load
		 * @param fragmentShader
		 *            the path of the fragment shader to load
		 * @return the openGL program index of the shader program
		 */
		static int generateShaderProgram(String vertexShader, String fragmentShader) {
			GL4 gl = (GL4) GLContext.getCurrentGL();

			int vShader = -1;
			if(vertexShader != null) {
				vShader = getShader(vertexShader, GL_VERTEX_SHADER);
			}
			
			int fShader = -1;
			if(fragmentShader != null) {
				fShader = getShader(fragmentShader, GL_FRAGMENT_SHADER);
			}
			
			if(vShader == -1 && fShader == -1) {
				System.out.println("Could not Load Shaders, exiting.");
				System.exit(-1);
			}

			// Link the compiled program
			System.out.println("Linking Shader Program...");
			int vfProgram = gl.glCreateProgram();
			if(vShader != -1) {
				gl.glAttachShader(vfProgram, vShader);
			}
			if(fShader != -1) {
				gl.glAttachShader(vfProgram, fShader);
			}

			gl.glLinkProgram(vfProgram);

			if (printProgramLog(vfProgram)) {
				System.out.println("Shader Linking Succeeded");
			} else {
				System.out.println("Shader Linking Failed");
			}

			if (checkOpenGLError()) {
				System.out.println("Open GL Errors found, exiting");
				System.exit(-1);
			}
			return vfProgram;
		}

		private static int getShader(String shaderPath, int shaderType) {
			if (shaderPath == null) {
				return -1;
			}
			
			String shaderTypeString = "";
			switch (shaderType) {
			case GL_VERTEX_SHADER:
				shaderTypeString = "Vertex";
				break;
			case GL_FRAGMENT_SHADER:
				shaderTypeString = "Fragment";
				break;
			}

			// Read in the shader sources
			String shaderSource[] = GLSLUtils.readShaderSource(shaderPath);

			// Check to see if the reads succeeded
			if (shaderSource == null) {
				System.exit(1);
			}

			// create the shader
			GL4 gl = (GL4) GLContext.getCurrentGL();
			int shader = gl.glCreateShader(shaderType);

			// Set the sources
			gl.glShaderSource(shader, shaderSource.length, shaderSource, null, 0);

			// Compile the shaders
			System.out.println("Compiling " + shaderTypeString + " Shader...");
			gl.glCompileShader(shader);


			if (printShaderLog(shader)) {
				System.out.println(shaderTypeString + " Shader Compilation Succeeded");
			} else {
				System.out.println(shaderTypeString + " Shader Compilation Failed");
			}
			
			return shader;
		}

		/**
		 * Modified return value from book example
		 *
		 * @param shader
		 *            the OpenGL shader index of the shader to print
		 * @return if the shader log contained anything or not
		 */
		static boolean printShaderLog(int shader) {
			GL4 gl = (GL4) GLContext.getCurrentGL();
			int[] len = new int[1];
			int[] chWrittn = new int[1];
			byte[] log = null;

			// determine the length of the shader compilation log
			gl.glGetShaderiv(shader, GL_INFO_LOG_LENGTH, len, 0);
			if (len[0] > 1) {
				log = new byte[len[0]];
				gl.glGetShaderInfoLog(shader, len[0], chWrittn, 0, log, 0);
				System.out.println("Shader Info Log: ");
				for (int i = 0; i < log.length; i++) {
					System.out.print((char) log[i]);
				}
				return false;
			}
			return true;
		}

		/**
		 * Modified return value from book example
		 *
		 * @param prog
		 *            the OpenGL program index whose program log to print
		 * @return if the program log contained anything or not
		 */
		static boolean printProgramLog(int prog) {
			GL4 gl = (GL4) GLContext.getCurrentGL();
			int[] len = new int[1];
			int[] chWrittn = new int[1];
			byte[] log = null;

			// determine length of the program compilation log
			gl.glGetProgramiv(prog, GL_INFO_LOG_LENGTH, len, 0);
			if (len[0] > 1) {
				log = new byte[len[0]];
				gl.glGetProgramInfoLog(prog, len[0], chWrittn, 0, log, 0);
				System.out.println("Program Info Log: ");
				for (int i = 0; i < log.length; i++) {
					System.out.print((char) log[i]);
				}
				return false;
			}
			return true;
		}

		/**
		 * @return if there is an open gl error or not
		 */
		static boolean checkOpenGLError() {
			GL4 gl = (GL4) GLContext.getCurrentGL();
			boolean foundError = false;
			GLU glu = new GLU();
			int glErr = gl.glGetError();
			while (glErr != GL_NO_ERROR) {
				System.err.println("glError: " + glu.gluErrorString(glErr));
				foundError = true;
				glErr = gl.glGetError();
			}
			return foundError;
		}

		static Texture loadTexture(String textureFileName) {
			Texture tex = null;
			try {
				tex = TextureIO.newTexture(new File(textureFileName), false);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return tex;
		}

		/**
		 * Copied verbatim from the book.
		 * 
		 * @author V. Scott Gordon
		 * @author John Clevenger
		 *
		 */
		static class ModelImporter {
			private ArrayList<Float> vertVals = new ArrayList<Float>();
			private ArrayList<Float> triangleVerts = new ArrayList<Float>();
			private ArrayList<Float> textureCoords = new ArrayList<Float>();
			private ArrayList<Float> stVals = new ArrayList<Float>();
			private ArrayList<Float> normals = new ArrayList<Float>();
			private ArrayList<Float> normVals = new ArrayList<Float>();

			public void parseOBJ(String filename) throws IOException {
				InputStream input = ModelImporter.class.getResourceAsStream(filename);
				BufferedReader br = new BufferedReader(new InputStreamReader(input));
				String line;
				while ((line = br.readLine()) != null) {
					if (line.startsWith("v ")) // vertex position ("v" case)
					{
						for (String s : (line.substring(2)).split(" ")) {
							vertVals.add(Float.valueOf(s));
						}
					} else if (line.startsWith("vt")) // texture coordinates ("vt" case)
					{
						for (String s : (line.substring(3)).split(" ")) {
							stVals.add(Float.valueOf(s));
						}
					} else if (line.startsWith("vn")) // vertex normals ("vn" case)
					{
						for (String s : (line.substring(3)).split(" ")) {
							normVals.add(Float.valueOf(s));
						}
					} else if (line.startsWith("f")) // triangle faces ("f" case)
					{
						for (String s : (line.substring(2)).split(" ")) {
							String v = s.split("/")[0];
							String vt = s.split("/")[1];
							String vn = s.split("/")[2];

							int vertRef = (Integer.valueOf(v) - 1) * 3;
							int tcRef = (Integer.valueOf(vt) - 1) * 2;
							int normRef = (Integer.valueOf(vn) - 1) * 3;

							triangleVerts.add(vertVals.get(vertRef));
							triangleVerts.add(vertVals.get((vertRef) + 1));
							triangleVerts.add(vertVals.get((vertRef) + 2));

							textureCoords.add(stVals.get(tcRef));
							textureCoords.add(stVals.get(tcRef + 1));

							normals.add(normVals.get(normRef));
							normals.add(normVals.get(normRef + 1));
							normals.add(normVals.get(normRef + 2));
						}
					}
				}
				input.close();
			}

			public int getNumVertices() {
				return (triangleVerts.size() / 3);
			}

			public float[] getVertices() {
				float[] p = new float[triangleVerts.size()];
				for (int i = 0; i < triangleVerts.size(); i++) {
					p[i] = triangleVerts.get(i);
				}
				return p;
			}

			public float[] getTextureCoordinates() {
				float[] t = new float[(textureCoords.size())];
				for (int i = 0; i < textureCoords.size(); i++) {
					t[i] = textureCoords.get(i);
				}
				return t;
			}

			public float[] getNormals() {
				float[] n = new float[(normals.size())];
				for (int i = 0; i < normals.size(); i++) {
					n[i] = normals.get(i);
				}
				return n;
			}
		}

	}

}