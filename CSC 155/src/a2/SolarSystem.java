package a2;

import a2.Primitives.Sphere;
import a2.Primitives.Tetrahedron;
import graphicslib3D.Point3D;
import graphicslib3D.Vector3D;

/**
 * The Model portion of the MVC architecture This contains all of the
 * information being acted on
 *
 * @author Graeme Knowles
 */
public class SolarSystem extends ModelObject {

	public SolarSystem() {
		this.setPreferredCameraPosition(new Point3D(0,0,-500));
	}

	@Override
	public void init() {
		// Orbital Speeds
		final double SEARTH = .1, SMERCURY = SEARTH * 1.607, SVENUS = SEARTH * 1.76, SMARS = SEARTH * 0.808,
				SJUPITER = SEARTH * 0.438, SSATURN = SEARTH * 0.325, SURANUS = SEARTH * 0.229, SNEPTUNE = SEARTH * .182;
		// Planet Diameters
		final double DEARTH = 1, DMERCURY = DEARTH * .38, DVENUS = DEARTH * .95, DMARS = DEARTH * .53,
				DJUPITER = DEARTH * 11.2, DSATURN = DEARTH * 9.45, DURANUS = DEARTH * 4, DNEPTUNE = DEARTH * 3.88;
		final double SOLAR_DIAMETER = 2 * DJUPITER;
		// Orbital Radii
		final double DISEARTH = SOLAR_DIAMETER + 50, DISMERCURY = DISEARTH * .387, DISVENUS = DISEARTH * .723, DISMARS = DISEARTH * 1.524,
				DISJUPITER = DISEARTH * 5.203, DISSATURN = DISEARTH * 9.537, DISURANUS = DISEARTH * 19.191, DISNEPTUNE = DISEARTH * 30.069;
		
		Sphere sunModel = new Primitives.Sphere(Util.Path.path + "/sunmap.jpg");
		SolarSystemObject theSun = new SolarSystemObject(sunModel, this);
		theSun.setScale(SOLAR_DIAMETER);
		theSun.setDailyRotMag(new Vector3D(0, -2, 0));

		Sphere mercuryModel = new Primitives.Sphere(Util.Path.path + "/squareMoonMap.jpg");
		SolarSystemObject mercury = new SolarSystemObject(mercuryModel, theSun);
		mercury.setScale(DMERCURY);
		mercury.setOrbitRadius(DISMERCURY);
		mercury.setOrbitalSpeed(SMERCURY);
		mercury.setDailyRotMag(new Vector3D(0, -SMERCURY * 3 / 2, 0));

		Sphere venusModel = new Primitives.Sphere(Util.Path.path + "/venus.jpg");
		SolarSystemObject venus = new SolarSystemObject(venusModel, theSun);
		venus.setScale(DVENUS);
		venus.setOrbitRadius(DISVENUS);
		venus.setOrbitalSpeed(SVENUS);
		venus.setDailyRotMag(new Vector3D(0, SVENUS / 1.92 , 0));
		
		Tetrahedron tetraMoonModel = new Tetrahedron(Util.Path.path + "/moon2.jpg");
		SolarSystemObject moonBase = new SolarSystemObject(tetraMoonModel, venus);
		moonBase.setScale(.25);
		moonBase.setOrbitRadius(3);
		moonBase.setOrbitalSpeed(-1);
		moonBase.setDailyRotMag(new Vector3D(1, 1, 1));

		Sphere earthModel = new Primitives.Sphere(Util.Path.path + "/earthmap1k.jpg");
		SolarSystemObject earth = new SolarSystemObject(earthModel, theSun);
		earth.setScale(DEARTH);
		earth.setOrbitRadius(DISEARTH);
		earth.setOrbitalSpeed(SEARTH);
		earth.setDailyRotMag(new Vector3D(0, -SEARTH * 365, 0));

		SolarSystemObject moon = new SolarSystemObject(mercuryModel, earth);
		moon.setScale(.25);
		moon.setOrbitRadius(5);
		moon.setOrbitalSpeed(SEARTH * 30);
		moon.setDailyRotMag(new Vector3D(0, -1, 0));

		Sphere marsModel = new Primitives.Sphere(Util.Path.path + "/mars.jpg");
		SolarSystemObject mars = new SolarSystemObject(marsModel, theSun);
		mars.setScale(DMARS);
		mars.setOrbitRadius(DISMARS);
		mars.setOrbitalSpeed(SMARS);
		mars.setDailyRotMag(new Vector3D(0, -SMARS * 686, 0));
		
		Sphere jupiterModel = new Primitives.Sphere(Util.Path.path + "/jupiter.jpg");
		SolarSystemObject jupiter = new SolarSystemObject(jupiterModel, theSun);
		jupiter.setScale(DJUPITER);
		jupiter.setOrbitRadius(DISJUPITER);
		jupiter.setOrbitalSpeed(SJUPITER);
		jupiter.setDailyRotMag(new Vector3D(0, -SJUPITER * 4332, 0));
		
		Sphere saturnModel = new Primitives.Sphere(Util.Path.path + "/saturn.jpg");
		SolarSystemObject saturn = new SolarSystemObject(saturnModel, theSun);
		saturn.setScale(DSATURN);
		saturn.setOrbitRadius(DISSATURN);
		saturn.setOrbitalSpeed(SSATURN);
		saturn.setDailyRotMag(new Vector3D(0, -SSATURN * 10759, 0));
		
		Sphere uranusModel = new Primitives.Sphere(Util.Path.path + "/uranus.jpg");
		SolarSystemObject uranus = new SolarSystemObject(uranusModel, theSun);
		uranus.setScale(DURANUS);
		uranus.setOrbitRadius(DISURANUS);
		uranus.setOrbitalSpeed(SURANUS);
		uranus.setDailyRotMag(new Vector3D(0, -SURANUS * 30688, 0));
		
		Sphere neptuneModel = new Primitives.Sphere(Util.Path.path + "/neptune.jpg");
		SolarSystemObject neptune = new SolarSystemObject(neptuneModel, theSun);
		neptune.setScale(DNEPTUNE);
		neptune.setOrbitRadius(DISNEPTUNE);
		neptune.setOrbitalSpeed(SNEPTUNE);
		neptune.setDailyRotMag(new Vector3D(0, -SNEPTUNE * 60182, 0));
	}
}
