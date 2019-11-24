#include "dubins.h"
#include <math.h>

int main(){

	// Initial data
	double x0 = 0;
	double y0 = 0;
	double th0 = -2/3*M_PI;
	double xf = 4;
	double yf = 0;
	double thf = M_PI/3.0;
	double kmax = 3.0;

	// Temporary s values
	double s1 = 1.0;	
	double s2 = 1.0;
	double s3 = 1.0;

	// Temporary k values
	double k0 =	3.0;
	double k1 =	3.0;
	double k2 =	3.0;

	//dubinsCurve curve;
	//set_dubinsCurve(curve,x0,y0,th0,s1,s2,s3,k0,k1,k2);
	//dubins_shortest_path(x0,y0,th0,xf,yf,thf,kmax);

	//cout << curve.arc_1.xf << "\n";
	//cout << curve.arc_2.x0 << "\n";
	//cout << curve.L << "\n";

	dubinsCurve curve;
	dubins_shortest_path(curve,x0,y0,th0,xf,yf,thf,kmax);

	return 0;
}






