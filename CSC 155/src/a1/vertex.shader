#version 430

uniform vec4 position;
uniform float size;
uniform vec4 vertex1Color;
uniform vec4 vertex2Color;
uniform vec4 vertex3Color;

out vec4 vertColor;

void main(void)
{ 
	float size2 = size/2;
	switch(gl_VertexID)
	{
		case 0:
			gl_Position = vec4(position[0] + size2, position[1] - size2, position[2] + size2, 1.0);
			vertColor = vertex1Color;
			break;
		case 1:
			gl_Position = vec4(position[0], position[1] + size2, position[2] + size2, 1.0);
  			vertColor = vertex2Color;
			break;
		case 2:
			gl_Position = vec4(position[0] - size2, position[1] - size2, position[2] + size2, 1.0);
  			vertColor = vertex3Color;
			break;
		default:
		break;
  	}
}