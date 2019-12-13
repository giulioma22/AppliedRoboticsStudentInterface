#ifndef DUBINS_HPP_
#define DUBINS_HPP_

#include <vector>
#include "utils.hpp"	// Path struct

double sinc(double t);

double mod2pi(double ang);

double rangeSymm(double const& ang);

bool check(double const& s1, double const& k0, double const& s2, double const& k1, double const& s3, double const& k2, double const& th0, double const& thf);

void scaleToStandard(double x0, double y0, double th0, double xf, double yf, double thf, double kmax, double& sc_th0, double& sc_thf, double& sc_kmax, double& lambda);

void scaleFromStandard(double lambda, double sc_s1, double sc_s2, double sc_s3, double& s1, double& s2, double& s3);

void LSL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

void RSR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

void LSR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

void RSL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

void RLR(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

void LRL(double sc_th0, double sc_thf, double sc_kmax, bool& ok, double& sc_s1, double& sc_s2, double& sc_s3);

// Structure for a Dubins arc
struct dubinsArc {
	double x0, y0, th0, k, s, xf, yf, thf;
};

// Structure for a Dubins curve
struct dubinsCurve {
	dubinsArc arc_1, arc_2, arc_3;
	double L;
};

void set_dubinsArc(dubinsArc& ptr, double x0, double y0, double th0, double k, double s);

void set_dubinsCurve(dubinsCurve& curve_ptr, double x0, double y0, double th0, double s1, double s2, double s3, double k0, double k1, double k2);

void dubins_shortest_path(dubinsCurve& curve, double const& x0, double const& y0, double const& th0, double const& xf, double const& yf, double const& thf, double const& kmax);

void discretize_arc(dubinsArc& full_arc, double& s, int& npts, Path& path);

#endif
