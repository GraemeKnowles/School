package a2;

import static com.jogamp.opengl.GL.GL_NO_ERROR;
import static com.jogamp.opengl.GL.GL_VERSION;
import static com.jogamp.opengl.GL2ES2.GL_FRAGMENT_SHADER;
import static com.jogamp.opengl.GL2ES2.GL_INFO_LOG_LENGTH;
import static com.jogamp.opengl.GL2ES2.GL_VERTEX_SHADER;

import java.io.File;

import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.glu.GLU;
import com.jogamp.opengl.util.texture.Texture;
import com.jogamp.opengl.util.texture.TextureIO;

import graphicslib3D.GLSLUtils;
import graphicslib3D.Matrix3D;

/**
 * Provides Utility/Helper functions for various aspects of the program
 *
 * @author Graeme Knowles
 */
class Util {

	static class Path{
		public static final String path = "src";
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

			// Read in the shader sources
			String vertexShaderSource[] = GLSLUtils.readShaderSource(vertexShader);
			String fragmentShaderSource[] = GLSLUtils.readShaderSource(fragmentShader);

			// Check to see if the reads succeeded
			if (vertexShaderSource == null || fragmentShaderSource == null) {
				System.exit(1);
			}

			// Create the shaders
			int vShader = gl.glCreateShader(GL_VERTEX_SHADER);
			int fShader = gl.glCreateShader(GL_FRAGMENT_SHADER);

			// Set the sources
			gl.glShaderSource(vShader, vertexShaderSource.length, vertexShaderSource, null, 0);
			gl.glShaderSource(fShader, fragmentShaderSource.length, fragmentShaderSource, null, 0);

			// Compile the shaders
			System.out.println("Compiling Vertex Shader...");
			gl.glCompileShader(vShader);
			if (printShaderLog(vShader)) {
				System.out.println("Vertex Shader Compilation Succeeded");
			} else {
				System.out.println("Vertex Shader Compilation Failed");
			}

			gl.glCompileShader(fShader);
			System.out.println("Compiling Fragment Shader...");
			if (printShaderLog(fShader)) {
				System.out.println("Fragment Shader Compilation Succeeded");
			} else {
				System.out.println("Fragment Shader Compilation Failed");
			}

			// Link the compiled program
			System.out.println("Linking Shader Program...");
			int vfProgram = gl.glCreateProgram();
			gl.glAttachShader(vfProgram, vShader);
			gl.glAttachShader(vfProgram, fShader);
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
	}
}