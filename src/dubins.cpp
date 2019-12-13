#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "dubins.h"
#include "utils.hpp"	// Path struct

using namespace std;

// Sinc() function
double sinc(double t){

	double s;
	if (abs(t) < 0.002){
		s = 1 - pow(t,2.0)/6.0*(1.0-pow(t,2.0)/20.0);
	} else {
		s = sin(t)/t;
	}

	return s;
}

// Normalize an angle (in range [0,2Pi])
double mod2pi(double ang){

	while (ang < 0){
		ang += 2*M_PI;
	}
	while (ang >= 2*M_PI){
		ang -= 2*M_PI;
	}

	return ang;
}

// Normalize angular difference
double rangeSymm(double const& ang){
	double out = ang;
	while (out < 0){
		out += 2*M_PI;
	}
	while (out >= 2*M_PI){
		out -= 2*M_PI;
	}
	return out;
}

bool check(double const& s1, double const& k0, double const& s2, double const& k1, double const& s3, double const& k2, double const& th0, double const& thf){
	int x0, y0, xf, yf;
	x0 = -1;
	y0 = 0;
	xf = 1;
	yf = 0;

	double eq1 = x0 + s1*sinc((1/2.0)*k0*s1)*cos(th0+(1/2.0)*k0*s1) + s2*sinc((1/2.0)*k1*s2)*cos(th0 + k0*s1 + (1/2.0)*k1*s2) + s3*sinc((1/2.0)*k2*s3)*cos(th0 + k0*s1 + k1*s2 + (1/2.0)*k2*s3) - xf;
	double eq2 = y0 + s1*sinc((1/2.0)*k0*s1)*sin(th0+(1/2.0)*k0*s1) + s2*sinc((1/2.0)*k1*s2)*sin(th0 + k0*s1 + (1/2.0)*k1*s2) + s3*sinc((1/2.0)*k2*s3)*sin(th0 + k0*s1 + k1*s2 + (1/2.0)*k2*s3) - yf;
	double eq3 =rangeSymm(x0*s1 + k1*s2 + k2*s3 + th0 + thf);

	bool bool_val = (sqrt(eq1*eq1+eq2*eq2+eq3*eq3) < 1e-10) and ((s1>0) or (s2>0) or (s3>0));

	return bool_val;
}

// Scale input problem to standard form
void scaleToStandard(double x0, double y0, double th0, double xf, double yf,
					double thf, double kmax, double& sc_th0, double& sc_thf, 
					double& sc_kmax, double& lambda){

	double dx = xf - x0;
	double dy = yf - y0;
	double phi = atan2(dy,dx);
	lambda = hypot(dx,dy)/2.0;

	sc_th0 = mod2pi(th0-phi);
	sc_thf = mod2pi(thf-phi);
	sc_kmax = kmax * lambda;
}

// Scale solution from standard problem to original
void scaleFromStandard(double lambda, double sc_s1, double sc_s2, double sc_s3, double& s1, double& s2, double& s3){
	
	s1 = sc_s1*lambda;
	s2 = sc_s2*lambda;
	s3 = sc_s3*lambda;	

}

// LSL
void LSL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_thf) - cos(sc_th0);
	double S = 2.0*sc_kmax + sin(sc_th0) - sin(sc_thf);
	double temp1 = atan2(C,S);
	sc_s1 = invK*mod2pi(temp1-sc_th0);
	double temp2 = 2.0+4.0*pow(sc_kmax,2)-2*cos(sc_th0-sc_thf)+4*sc_kmax*(sin(sc_th0)-sin(sc_thf));
	if (temp2 < 0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*sqrt(temp2);
	sc_s3 = invK*mod2pi(sc_thf-temp1);
	ok = true;
}

// RSR
void RSR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_th0) - cos(sc_thf);
	double S = 2.0*sc_kmax - sin(sc_th0) + sin(sc_thf);
	double temp1 = atan2(C,S);
	sc_s1 = invK*mod2pi(sc_th0-temp1);
	double temp2 = 2.0+4.0*pow(sc_kmax,2)-2*cos(sc_th0-sc_thf)-4*sc_kmax*(sin(sc_th0)-sin(sc_thf));
	if (temp2 < 0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*sqrt(temp2);
	sc_s3 = invK*mod2pi(temp1-sc_thf);
	ok = true;
}

// LSR
void LSR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_th0) + cos(sc_thf);
	double S = 2.0*sc_kmax + sin(sc_th0) + sin(sc_thf);
	double temp1 = atan2(-C,S);
	double temp3 = 4.0*pow(sc_kmax,2)-2.0+2.0*cos(sc_th0-sc_thf)+4.0*sc_kmax*(sin(sc_th0)+sin(sc_thf));

	if (temp3 < 0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*sqrt(temp3);
	double temp2 = -atan2(-2.0,sc_s2*sc_kmax);
	sc_s1 = invK*mod2pi(temp1+temp2-sc_th0);
	sc_s3 = invK*mod2pi(temp1+temp2-sc_thf);
	ok = true;

}

// RSL
void RSL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_th0) + cos(sc_thf);
	double S = 2.0*sc_kmax - sin(sc_th0) - sin(sc_thf);
	double temp1 = atan2(C,S);
	double temp3 = 4.0*pow(sc_kmax,2)-2.0+2.0*cos(sc_th0-sc_thf)-4.0*sc_kmax*(sin(sc_th0)+sin(sc_thf));
	if (temp3 < 0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*sqrt(temp3);
	double temp2 = atan2(2.0,sc_s2*sc_kmax);
	sc_s1 = invK*mod2pi(sc_th0-temp1+temp2);
	sc_s3 = invK*mod2pi(sc_thf-temp1+temp2);
	ok = true;
}

// RLR
void RLR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_th0) - cos(sc_thf);
	double S = 2.0*sc_kmax - sin(sc_th0) + sin(sc_thf);
	double temp1 = atan2(C,S);
	double temp2 = 0.125*(6.0-4.0*pow(sc_kmax,2)+2.0*cos(sc_th0-sc_thf)+4.0*sc_kmax*(sin(sc_th0)-sin(sc_thf)));
	if (abs(temp2) > 1.0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*mod2pi(2.0*M_PI-acos(temp2));
	sc_s1 = invK*mod2pi(sc_th0-temp1+0.5*sc_s2*sc_kmax);
	sc_s3 = invK*mod2pi(sc_th0-sc_thf+sc_kmax*(sc_s2-sc_s1));
	ok = true;
}

// LRL
void LRL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3){
	double invK = 1.0/sc_kmax;
	double C = cos(sc_thf) - cos(sc_th0);
	double S = 2.0*sc_kmax + sin(sc_th0) - sin(sc_thf);
	double temp1 = atan2(C,S);
	double temp2 = 0.125*(6.0-4.0*pow(sc_kmax,2)+2.0*cos(sc_th0-sc_thf)+4.0*sc_kmax*(sin(sc_th0)-sin(sc_thf)));
	if (abs(temp2) > 1.0){
		ok = false;
		sc_s1 = 0;
		sc_s2 = 0;
		sc_s3 = 0;
		return;
	}
	sc_s2 = invK*mod2pi(2*M_PI-acos(temp2));
	sc_s1 = invK*mod2pi(temp1-sc_th0+0.5*sc_s2*sc_kmax);
	sc_s3 = invK*mod2pi(sc_thf-sc_th0+sc_kmax*(sc_s2-sc_s1));
	ok = true;
}

// Set a structure representing a Dubins arc (straight or circular)
void set_dubinsArc(dubinsArc& ptr, double x0, double y0, double th0, double k, double s){
	
	ptr.x0 = x0;
	ptr.y0 = y0;
	ptr.th0 = th0;
	ptr.k = k;
	ptr.s = s;
	ptr.xf = x0 + s * sinc(k*s/2.0) * cos(th0+k*s/2.0);
	ptr.yf = y0 + s * sinc(k*s/2.0) * sin(th0+k*s/2.0);
	ptr.thf = mod2pi(th0+k*s);

}

// Set a structure representing a Dubins curve (composed by 3 arcs)
void set_dubinsCurve(dubinsCurve& curve_ptr, double x0, double y0, double th0, double s1, double s2, double s3, double k0, double k1, double k2){

	set_dubinsArc(curve_ptr.arc_1, x0, y0, th0, k0, s1);
	set_dubinsArc(curve_ptr.arc_2, curve_ptr.arc_1.xf, 
				curve_ptr.arc_1.yf, curve_ptr.arc_1.thf, k1, s2);
	set_dubinsArc(curve_ptr.arc_3, curve_ptr.arc_2.xf, 
				curve_ptr.arc_2.yf, curve_ptr.arc_2.thf, k2, s3);
	curve_ptr.L = curve_ptr.arc_1.s + curve_ptr.arc_2.s + 
				curve_ptr.arc_3.s;
}

void dubins_shortest_path(dubinsCurve& curve, double const& x0, double const& y0, double const& th0, double const& xf, double const& yf, double const& thf, double const& kmax){

	double sc_th0, sc_thf, sc_kmax, lambda;
	scaleToStandard(x0, y0, th0, xf, yf, thf, kmax, sc_th0, sc_thf, 
					sc_kmax, lambda);
	
	// Create typedef for recognizing pointers to function
	typedef void (*type0)(double,double,double,bool&,double&,double&,double&);

	void(*LSL_ptr)(double,double,double,bool&,double&,double&,double&)= &LSL;
	void(*RSR_ptr)(double,double,double,bool&,double&,double&,double&)= &RSR;
	void(*LSR_ptr)(double,double,double,bool&,double&,double&,double&)= &LSR;
	void(*RSL_ptr)(double,double,double,bool&,double&,double&,double&)= &RSL;
	void(*RLR_ptr)(double,double,double,bool&,double&,double&,double&)= &RLR;
	void(*LRL_ptr)(double,double,double,bool&,double&,double&,double&)= &LRL;
	
	type0 primitives[6] = {LSL_ptr,RSR_ptr,LSR_ptr,RSL_ptr,RLR_ptr,LRL_ptr}; 
	int ksigns[6][3] = {{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},{-1,1,-1},{1,-1,1}};

	int pidx = -1.0;
	int L = 1e9;	// Infinite value
	bool ok;
	double s1, s2, s3;
	double sc_s1, sc_s2, sc_s3;
	double sc_s1_c, sc_s2_c, sc_s3_c;
	int Lcur;

	for (int i = 0; i < 6; i++){
		primitives[i](sc_th0, sc_thf, sc_kmax, ok, sc_s1_c, sc_s2_c, sc_s3_c);

		Lcur = sc_s1_c + sc_s2_c + sc_s3_c;
		if (ok and Lcur < L){
			L = Lcur;
			sc_s1 = sc_s1_c;
			sc_s2 = sc_s2_c;
			sc_s3 = sc_s3_c;
			pidx = i;
		}
	}

	if (pidx >= 0){
		// Transform problem to standard form
		scaleFromStandard(lambda, sc_s1, sc_s2, sc_s3, s1, s2, s3);

		// Construct Dubins curve		
		set_dubinsCurve(curve, x0, y0, th0, s1, s2, s3, ksigns[pidx][0]*kmax, ksigns[pidx][1]*kmax, ksigns[pidx][2]*kmax);
	}

}

// Discretize every arc to smaller arcs for the robot to follow

void discretize_arc(dubinsArc& full_arc, double& s, int& npts, Path& path){

	for (int i = 0; i <= npts; i++){
		dubinsArc small_arc;
		double s_local = full_arc.s/npts*i;
		set_dubinsArc(small_arc, full_arc.x0, full_arc.y0, full_arc.th0, full_arc.k, s_local);

		path.points.emplace_back(s_local, small_arc.xf, small_arc.yf, small_arc.thf, small_arc.k);

		s += full_arc.s/npts;

	}

}

