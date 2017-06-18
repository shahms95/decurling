#include <pngwriter.h>

using namespace std;
int main()

{

	pngwriter image(2550, 3300, 1.0, "out0.png");
	// int colours[] = {6553,6553,0};
	int colours[] = {6553,0,6553};
	image.plot(30, 40, 1.0, 0.0, 0.0);
	int xcentre = 255;
	int ycentre = 330;
	int radius = 20;
	double opacity = 1;
	int red = colours[0];
	int green = colours[1];
	int blue = colours[2];
	for (int i = 1; i <= 9; ++i)
	{
		for (int j = 1; j <= 9; ++j)
		{
			image.filledcircle((i)*xcentre, (j)*ycentre, radius, red*i, green, blue*j);
		}
	}
	image.close();



	return 0;

}