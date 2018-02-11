package a3;

import a3.Primitives.Sphere;
import a3.Util.JOGL.ModelImporter;
import graphicslib3D.Material;
import graphicslib3D.Point3D;
import graphicslib3D.light.AmbientLight;
import graphicslib3D.light.Light;
import graphicslib3D.light.PositionalLight;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import a3.Primitives.Pyramid;

public class A3Scene extends Scene {

	@Override
	public void init() {
		// Set up the models
		ModelObject sphereObj = new ModelObject(new Sphere(Util.Path.path + "/jupiter.jpg"), this);
		sphereObj.translate(1, 1, 1);
		sphereObj.setMaterial(Material.GOLD);

		ModelObject shuttleObj = new ModelObject(new Model3D("/shuttle.obj", Util.Path.path + "/spstob_1.jpg"), this);
		Material shuttleMat = new Material();
		shuttleMat.setShininess(.25f);
		shuttleMat.setDiffuse(new float[] { .55f, .55f, .55f, 1 });
		shuttleMat.setSpecular(new float[] { .7f, .7f, .7f, 1 });
		shuttleObj.setMaterial(shuttleMat);
		
		// Set up the lighting
		AmbientLight globalAmbient = AmbientLight.getAmbientLight();
		globalAmbient.setRed(.7f);
		globalAmbient.setGreen(.7f);
		globalAmbient.setBlue(.7f);
		globalAmbient.setAlpha(1.0f);

		PositionLight light = new PositionLight();
		light.translate(-1.5f, -1.5f, -1.5f);
		this.setLight(light);

		// Camera start position
		this.setPreferredCameraPosition(new Point3D(5f, 0f, 0f));
		this.setPreferredCameraRotation(new Point3D(0f, -90f, 0f));
	}
}
