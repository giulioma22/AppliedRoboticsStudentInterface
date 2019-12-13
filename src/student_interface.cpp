#include "student_image_elab_interface.hpp"
#include "student_planning_interface.hpp"

#include <stdexcept>
#include <sstream>

#include <vector>
#include <atomic>
#include <unistd.h>
#include <algorithm>

#include <experimental/filesystem>

#include "dubins.h"
#include "clipper/cpp/clipper.hpp"

namespace student {

bool sort_pair(const std::pair<int,Polygon>& a, const std::pair<int,Polygon>& b){
	return (a.first < b.first);
}

void loadImage(cv::Mat& img_out, const std::string& config_folder){  
  static bool initialized = false;
  static std::vector<cv::String> img_list; // list of images to load
  static size_t idx = 0;  // idx of the current img
  static size_t function_call_counter = 0;  // idx of the current img
  const static size_t freeze_img_n_step = 30; // hold the current image for n iteration
  static cv::Mat current_img; // store the image for a period, avoid to load it from file every time
  
  if(!initialized){
    const bool recursive = false;
    // Load the list of jpg image contained in the config_folder/img_to_load/
    cv::glob(config_folder + "/img_to_load/*.jpg", img_list, recursive);
    
    if(img_list.size() > 0){
      initialized = true;
      idx = 0;
      current_img = cv::imread(img_list[idx]);
      function_call_counter = 0;
    }else{
      initialized = false;
    }
  }
  
  if(!initialized){
    throw std::logic_error( "Load Image can not find any jpg image in: " +  config_folder + "/img_to_load/");
    return;
  }
  
  img_out = current_img;
  function_call_counter++;  
  
  // If the function is called more than N times load increment image idx
  if(function_call_counter > freeze_img_n_step){
    function_call_counter = 0;
    idx = (idx + 1)%img_list.size();    
    current_img = cv::imread(img_list[idx]);
  }
 }

static int i;
static bool state = false;

 void genericImageListener(const cv::Mat& img_in, std::string topic, const std::string& config_folder){
  	
	if (!state) {
		i = 0;
		state = true;
	}

	//CREATES ONE FOLDER IF DOESN'T EXISTS
	namespace fs = std::experimental::filesystem;
	std::stringstream src;
	src << config_folder << "saved_images/";


	if (!fs::is_directory(src.str()) || !fs::exists(src.str())) { 
	    fs::create_directory(src.str()); 
	}

	//SAVES IMAGE WHEN PRESS S ON THE KEYBOARD

	cv::imshow(topic, img_in);
	char k;
	k = cv::waitKey(30);

    	std::stringstream image;
    	
    	switch (k) {    	
		case 's':
				
			image << src.str() << std::setfill('0') << std::setw(4) << (i++) << ".jpg";
			std::cout << image.str() << std::endl;
			cv::imwrite(image.str(), img_in);

			std::cout << "The image" << image.str() << "was saved." << std::endl;
		 	break;
		default:
				break;
	} 

  }


// Function to pick arena points - - - - - - -

  static cv::Mat bg_img;
  static std::vector<cv::Point2f> result;
  static std::string name;
  static std::atomic<bool> done;
  static int n;
  static double show_scale = 2.0;

  void mouseCallback(int event, int x, int y, int, void* p)
  {
    if (event != cv::EVENT_LBUTTONDOWN || done.load()) return;
    
    result.emplace_back(x*show_scale, y*show_scale);
    cv::circle(bg_img, cv::Point(x,y), 20/show_scale, cv::Scalar(0,0,255), -1);
    cv::imshow(name.c_str(), bg_img);

    if (result.size() >= n) {
      usleep(500*1000);
      done.store(true);
    }
  }

  std::vector<cv::Point2f> pickNPoints(int n0, const cv::Mat& img)
  {
    result.clear();
    cv::Size small_size(img.cols/show_scale, img.rows/show_scale);
    cv::resize(img, bg_img, small_size);
    //bg_img = img.clone();
    name = "Pick " + std::to_string(n0) + " points";
    cv::imshow(name.c_str(), bg_img);
    cv::namedWindow(name.c_str());
    n = n0;

    done.store(false);

    cv::setMouseCallback(name.c_str(), &mouseCallback, nullptr);
    while (!done.load()) {
      cv::waitKey(500);
    }

    cv::destroyWindow(name.c_str());
    return result;
  }

// - - - - - - - - - 


  bool extrinsicCalib(const cv::Mat& img_in, std::vector<cv::Point3f> object_points, const cv::Mat& camera_matrix, cv::Mat& rvec, cv::Mat& tvec, const std::string& config_folder){

    std::string extrinsic_path = config_folder + "extrinsicCalib.csv";
    std::vector<cv::Point2f> imagePoints;

  if (!std::experimental::filesystem::exists(extrinsic_path)){
          
    std::experimental::filesystem::create_directories(config_folder);
    imagePoints = pickNPoints(4, img_in);
    std::ofstream output(extrinsic_path);

      if (!output.is_open()){
        throw std::runtime_error("Cannot write file: " + extrinsic_path);
      }
      for (const auto pt: imagePoints) {
        output << pt.x << " " << pt.y << std::endl;
      }
      output.close();
  }else{
      std::ifstream input_file(extrinsic_path);

      while (!input_file.eof()){
        double x, y;
        if (!(input_file >> x >> y)) {
          if (input_file.eof()) break;
          else {
            throw std::runtime_error("Malformed file: " + extrinsic_path);
          }
        }
        imagePoints.emplace_back(x, y);
      }
      input_file.close();
  }

    bool result = cv::solvePnP(object_points, imagePoints, camera_matrix, {}, rvec, tvec);

    return result;

  }

  void imageUndistort(const cv::Mat& img_in, cv::Mat& img_out, 
          const cv::Mat& cam_matrix, const cv::Mat& dist_coeffs, const std::string& config_folder){

    cv::undistort(img_in, img_out, cam_matrix, dist_coeffs);

  }
 
  void findPlaneTransform(const cv::Mat& cam_matrix, const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point3f>& object_points_plane, const std::vector<cv::Point2f>& dest_image_points_plane, cv::Mat& plane_transf, const std::string& config_folder){
    cv::Mat image_points;
    // projectPoint output is image_points
    cv::projectPoints(object_points_plane, rvec, tvec, cam_matrix, cv::Mat(), image_points);
    plane_transf = cv::getPerspectiveTransform(image_points, dest_image_points_plane);
  }

  void unwarp(const cv::Mat& img_in, cv::Mat& img_out, const cv::Mat& transf, const std::string& config_folder){
    cv::warpPerspective(img_in, img_out, transf, img_in.size());
  }


  bool detect_red(const cv::Mat& hsv_img, const double scale, std::vector<Polygon>& obstacle_list){
    
    cv::Mat red_mask_low, red_mask_high, red_mask;
    cv::inRange(hsv_img, cv::Scalar(0, 72, 105), cv::Scalar(20, 255, 255), red_mask_low);
    //cv::inRange(hsv_img, cv::Scalar(175, 10, 10), cv::Scalar(179, 255, 255), red_mask_high);
    cv::inRange(hsv_img, cv::Scalar(130, 81, 49), cv::Scalar(180, 255, 150), red_mask_high);
    cv::addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0, red_mask);
    
    // Find red regions
    std::vector<std::vector<cv::Point>> contours, contours_approx;
    std::vector<cv::Point> approx_curve;
    // Process red mask
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (int i=0; i<contours.size(); ++i)
    {
      // Approximate polygon w/ fewer vertices if not precise
      // 3rd arg - max distance original curve to approx
      approxPolyDP(contours[i], approx_curve, 3, true);

      Polygon scaled_contour;
      for (const auto& pt: approx_curve) {
        scaled_contour.emplace_back(pt.x/scale, pt.y/scale);
      }
      // Add obstacle to list
      obstacle_list.push_back(scaled_contour);
    }

	std::vector<Polygon> inflated_obstacles;

	const double INT_ROUND = 1000.;

	for (int obs = 0; obs < obstacle_list.size(); ++obs) {

		ClipperLib::Path srcPoly;
		ClipperLib::Paths newPoly;
		ClipperLib::ClipperOffset co;

		for (int ver = 0; ver < obstacle_list[obs].size(); ++ver){
			int x = obstacle_list[obs][ver].x * INT_ROUND;
			int y = obstacle_list[obs][ver].y * INT_ROUND;
			srcPoly << ClipperLib::IntPoint(x,y);
		}


		co.AddPath(srcPoly, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);

		co.Execute(newPoly, 50);	// Int is obstacle inflation idx

		for (const ClipperLib::Path &path: newPoly){
			// Obstacle obst = create data structure for current obstacle...
			Polygon obst;
			for (const ClipperLib::IntPoint &pt: path){
				double x = pt.X / INT_ROUND;
				double y = pt.Y / INT_ROUND;
				// Add vertex (x,y) to current obstacle...
				obst.emplace_back(x,y); 
			}
			// Close and export current obstacle...
  			inflated_obstacles.push_back(obst);
			obstacle_list[obs] = obst;
		}
	
	}

    return true;

  }

  bool detect_green_gate(const cv::Mat& hsv_img, const double scale, Polygon& gate){

    cv::Mat green_mask_gate;    
    //cv::inRange(hsv_img, cv::Scalar(50, 80, 34), cv::Scalar(75, 255, 255), green_mask_gate);
    //cv::inRange(hsv_img, cv::Scalar(13, 68, 41), cv::Scalar(86, 255, 80), green_mask_gate);
	cv::inRange(hsv_img, cv::Scalar(15, 65, 40), cv::Scalar(85, 255, 95), green_mask_gate);
    
    // Find green regions - GATE
    std::vector<std::vector<cv::Point>> contours, contours_approx;
    std::vector<cv::Point> approx_curve;
    // Process green mask - GATE
    cv::findContours(green_mask_gate, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    bool gate_found = false;

    /*for(auto& contour : contours){
      const double area = cv::contourArea(contour);
      if (area > 500){
        // Approximate polygon w/ fewer vertices if not precise
        approxPolyDP(contour, approx_curve, 3, true);

        for (const auto& pt: approx_curve) {
          // Store (scaled) values of gate
          gate.emplace_back(pt.x/scale, pt.y/scale);
        }
        gate_found = true;
        break;
      }      
    }*/

    for(auto& contour : contours){
      // Approximate polygon w/ fewer vertices if not precise
      approxPolyDP(contour, approx_curve, 30, true);
      if (approx_curve.size() != 4) continue;
      for (const auto& pt: approx_curve) {
        // Store (scaled) values of gate
        gate.emplace_back(pt.x/scale, pt.y/scale);
      }
      gate_found = true;
      break;
    }
    
    return gate_found;
  }

cv::Mat rotate(cv::Mat in_ROI, double ang_degrees){
    cv::Mat out_ROI;
    cv::Point2f center(in_ROI.cols/2., in_ROI.rows/2.);  
 
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, ang_degrees, 1.0);

    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), in_ROI.size(), ang_degrees).boundingRect2f();
    
    rot_mat.at<double>(0,2) += bbox.width/2.0 - in_ROI.cols/2.0;
    rot_mat.at<double>(1,2) += bbox.height/2.0 - in_ROI.rows/2.0;
    
    warpAffine(in_ROI, out_ROI, rot_mat, bbox.size());
    return out_ROI;
  }
  
  const double MIN_AREA_SIZE = 100;
  std::string template_folder = "/home/lar2019/workspace/project/template/";
  bool detect_green_victims(const cv::Mat& hsv_img, const double scale, std::vector<std::pair<int,Polygon>>& victim_list){
    
    // Find green regions
    cv::Mat green_mask_victims;
    
    // store a binary image in green_mask where the white pixel are those contained in HSV rage (x,x,x) --> (y,y,y)
    //cv::inRange(hsv_img, cv::Scalar(50, 80, 34), cv::Scalar(75, 255, 255), green_mask_victims); //Simulator
    //cv::inRange(hsv_img, cv::Scalar(13, 68, 41), cv::Scalar(86, 255, 80), green_mask_victims);
	cv::inRange(hsv_img, cv::Scalar(15, 65, 40), cv::Scalar(85, 255, 95), green_mask_victims);

    // Apply some filtering
    // Create the kernel of the filter i.e. a rectanble with dimension 3x3
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size((1*2) + 1, (1*2)+1));
    // Dilate using the generated kernel
    cv::dilate(green_mask_victims, green_mask_victims, kernel);
    // Erode using the generated kernel
    cv::erode(green_mask_victims,  green_mask_victims, kernel);

    // Find green contours
    std::vector<std::vector<cv::Point>> contours, contours_approx;    
    // Create an image which we can modify not changing the original image
    cv::Mat contours_img;
    contours_img = hsv_img.clone();

    // Finds green contours in a binary (new) image
    cv::findContours(green_mask_victims, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); 

    // create an array of rectangle (i.e. bounding box containing the green area contour)  
    std::vector<cv::Rect> boundRect(contours.size());
    int victim_id = 0;
    for (int i=0; i<contours.size(); ++i){
      double area = cv::contourArea(contours[i]);
      if (area < MIN_AREA_SIZE) continue; // filter too small contours to remove false positives

      std::vector<cv::Point> approx_curve;/////////////////////////////////////////
      approxPolyDP(contours[i], approx_curve, 10, true);
      if(approx_curve.size() < 6) continue; //fitler out the gate 
     
      Polygon scaled_contour;
      for (const auto& pt: approx_curve) {
        scaled_contour.emplace_back(pt.x/scale, pt.y/scale);
      }
      // Add victims to the victim_list
      victim_list.push_back({victim_id++, scaled_contour}); 

      contours_approx = {approx_curve};
      // Draw the contours on image with a line color of BGR=(0,170,220) and a width of 3
      drawContours(contours_img, contours_approx, -1, cv::Scalar(0,170,220), 3, cv::LINE_AA);

      // find the bounding box of the green blob approx curve
      boundRect[i] = boundingRect(cv::Mat(approx_curve)); 
    }

    cv::Mat green_mask_victims_inv;

    // Init a matrix specify its dimension (img.rows, img.cols), default color(255,255,255) and elemet type (CV_8UC3).
    cv::Mat filtered(hsv_img.rows, hsv_img.cols, CV_8UC3, cv::Scalar(255,255,255));

    // generate binary mask with inverted pixels w.r.t. green mask -> black numbers are part of this mask
    cv::bitwise_not(green_mask_victims, green_mask_victims_inv); 

    // Load digits template images
    std::vector<cv::Mat> templROIs;
    for (int i=1; i<=5; ++i) {
      auto num_template = cv::imread(template_folder + std::to_string(i) + ".png");
      // mirror the template, we want them to have the same shape of the number that we have in the unwarped ground image
      cv::flip(num_template, num_template, 1); 

      // Store the template in templROIs (vector of mat)
      templROIs.emplace_back(num_template);
    }  
  
    // create copy of image without green shapes
    hsv_img.copyTo(filtered, green_mask_victims_inv);

    // create a 3x3 recttangular kernel for img filtering
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size((2*2) + 1, (2*2)+1));
  
    // For each green blob in the original image containing a digit
    int victim_counter = -1;
    for (int i=0; i<boundRect.size(); ++i){
      // Constructor of mat, we pass the original image and the coordinate to copy and we obtain an image pointing to that subimage
      cv::Mat processROI(filtered, boundRect[i]); // extract the ROI containing the digit

      if (processROI.empty()) continue;
      victim_counter = victim_counter+1;
      //std::cout << "MY INDEX: " << victim_counter << std::endl;   
      // The size of the number in the Template image should be similar to the dimension
      // of the number in the ROI
      cv::resize(processROI, processROI, cv::Size(200, 200)); // resize the ROI 
      cv::threshold( processROI, processROI, 100, 255, 0 );   // threshold and binarize the image, to suppress some noise
   
      // Apply some additional smoothing and filtering
      cv::erode(processROI, processROI, kernel);
      cv::GaussianBlur(processROI, processROI, cv::Size(5, 5), 2, 2);
      cv::erode(processROI, processROI, kernel);

      // Find the template digit with the best matching
      double maxScore = 0;
      int maxIdx = -1;
      cv::Mat rot_processROI(filtered, boundRect[i]);
      for(int k=0;k<36;++k){
        //Rotate processROI
        rot_processROI = rotate(processROI, 10*k);
        
        for (int j=0; j<templROIs.size(); ++j) {
          cv::Mat result;

          // Match the ROI with the templROIs j-th
          cv::matchTemplate(rot_processROI, templROIs[j], result, cv::TM_CCOEFF);
          double score;
          cv::minMaxLoc(result, nullptr, &score); 

          // Compare the score with the others, if it is higher save this as the best match!
          if (score > maxScore) {
            maxScore = score;
            maxIdx = j;

            //cv::imshow("ROI", rot_processROI);
          }
        }
      }
      victim_list.at(victim_counter).first = maxIdx + 1;
      // Display the best fitting number 
      //std::cout << "Best fitting template: " << maxIdx + 1 << std::endl; 
      //cv::waitKey(0);
    }

	sort(victim_list.begin(), victim_list.end(), sort_pair);

	std::cout << "\n\n - - - SUCCESSFUL DIGIT RECOGNITION - - - \n\n\n";
    return true; 
  }

  bool detect_blue_robot(const cv::Mat& hsv_img, const double scale, Polygon& triangle, double& x, double& y, double& theta){

    cv::Mat blue_mask;    
    //cv::inRange(hsv_img, cv::Scalar(92, 80, 50), cv::Scalar(145, 255, 255), blue_mask);
    //cv::inRange(hsv_img, cv::Scalar(200, 80, 20), cv::Scalar(220, 220, 225), blue_mask);
	cv::inRange(hsv_img, cv::Scalar(100, 75, 45), cv::Scalar(145, 255, 225), blue_mask);


    // Process blue mask
    std::vector<std::vector<cv::Point>> contours, contours_approx;
    std::vector<cv::Point> approx_curve;
    // Find the contours of blue objects
    cv::findContours(blue_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool robot_found = false;
    for (int i=0; i<contours.size(); ++i)
    {
      // Approximate polygon w/ fewer vertices if not precise
      cv::approxPolyDP(contours[i], approx_curve, 30, true);
      if (approx_curve.size() != 3) continue;
      robot_found = true;
      break;
    }

    if (robot_found) 
    {
      // Store values in triangle
      for (const auto& pt: approx_curve) {
        triangle.emplace_back(pt.x/scale, pt.y/scale);
      }

      // Find center of robot
      double cx, cy;
      for (auto item: triangle) 
      {
        cx += item.x;
        cy += item.y;
      }
      cx /= triangle.size();
      cy /= triangle.size();

      // Find further vertix from center
      double dst = 0;
      Point vertex;
      for (auto& item: triangle)
      {
        double dx = item.x-cx;      
        double dy = item.y-cy;
        double curr_d = dx*dx + dy*dy;
        if (curr_d > dst)
        { 
          dst = curr_d;
          vertex = item;
        }
      }

      // Calculate yaw
      double dx = cx-vertex.x;
      double dy = cy-vertex.y;

      // Robot position
      x = cx;
      y = cy;
      theta = std::atan2(dy, dx);
    }

    return robot_found;

}

  bool processMap(const cv::Mat& img_in, const double scale, std::vector<Polygon>& obstacle_list, std::vector<std::pair<int,Polygon>>& victim_list, Polygon& gate, const std::string& config_folder){
    cv::Mat hsv_img;
    cv::cvtColor(img_in, hsv_img, cv::COLOR_BGR2HSV);
    
    // Check if objects found
    const bool red = detect_red(hsv_img, scale, obstacle_list);
    if(!red) std::cout << "detect_red returns false" << std::endl;
    const bool green_gate = detect_green_gate(hsv_img, scale, gate);
    if(!green_gate) std::cout << "detect_green_gate returns false" << std::endl;
    const bool green_victims = detect_green_victims(hsv_img, scale, victim_list);
    if(!green_victims) std::cout << "detect_green_victims returns false" << std::endl;

    return red && green_gate && green_victims;
  }

  bool findRobot(const cv::Mat& img_in, const double scale, Polygon& triangle, double& x, double& y, double& theta, const std::string& config_folder){
    cv::Mat hsv_img;
    cv::cvtColor(img_in, hsv_img, cv::COLOR_BGR2HSV);
    // RGB to HSV to then detect blue of robot more easily
    return detect_blue_robot(hsv_img, scale, triangle, x, y, theta);    
  }

  bool planPath(const Polygon& borders, const std::vector<Polygon>& obstacle_list, const std::vector<std::pair<int,Polygon>>& victim_list, const Polygon& gate, const float x, const float y, const float theta, Path& path, const std::string& config_folder){



	int kmax = 10;		// Max angle of curvature
	int npts = 100;		// Standard discretization unit of arcs 
	dubinsCurve newDubins;

	// - - - GATE CENTER - - -

	double gateX = (gate[0].x + gate[1].x + gate[2].x + gate[3].x)/4;
	double gateY = (gate[0].y + gate[1].y + gate[2].y + gate[3].y)/4;

	// Compute gate orientation
	double gateTh;
	double dist_1 = sqrt(pow(gate[0].x-gate[1].x,2.0)+pow(gate[0].y-gate[1].y,2.0));
	double dist_2 = sqrt(pow(gate[1].x-gate[2].x,2.0)+pow(gate[1].y-gate[2].y,2.0));
	if (dist_1 < dist_2){
		gateTh = acos(abs(gate[0].x-gate[1].x) / dist_1);
	} else {
		gateTh = acos(abs(gate[1].x-gate[2].x) / dist_1);
	}

	//  - - - VICTIM CENTER - - -

	std::vector<Point> victim_center;
	double victim_X;
	double victim_Y;

	for (int i = 0; i < victim_list.size(); i++){
		victim_X = 0;
		victim_Y = 0;
		Polygon currentPoly = std::get<1>(victim_list[i]);
		for (int pt = 0; pt < currentPoly.size(); pt++){
			victim_X += currentPoly[pt].x;
			victim_Y += currentPoly[pt].y;
		}
		victim_X /= currentPoly.size();
		victim_Y /= currentPoly.size();
		
		victim_center.emplace_back(victim_X, victim_Y);

	}

	/* // Victim centers
	0.718819 ~ 0.219375
	0.96135 ~ 0.208894
	1.22356 ~ 0.20787
	1.25629 ~ 0.448013
	1.44934 ~ 0.685181
	*/

	// - - - DISCRETIZATION - - - 

	double s = 0;

	/*for (int a = 0; a <= victim_center.size(); a++){

		if (a == 0) {
			dubins_shortest_path(newDubins, x, y, theta, victim_center[a].x, victim_center[a].y, M_PI/4, kmax);
		} else if (a == victim_center.size()){
			dubins_shortest_path(newDubins, path.points.back().x, path.points.back().y, path.points.back().theta, gateX, gateY, gateTh, kmax);		// If last, go to gate
		} else {
			dubins_shortest_path(newDubins, path.points.back().x, path.points.back().y, path.points.back().theta, victim_center[a].x, victim_center[a].y, M_PI/4, kmax);
		}

		discretize_arc(newDubins.arc_1, npts, path);	// Arc 1
		discretize_arc(newDubins.arc_2, npts, path);	// Arc 2
		discretize_arc(newDubins.arc_3, npts, path);	// Arc 3
	}*/


	dubins_shortest_path(newDubins, x, y, theta, victim_center[0].x, victim_center[0].y, M_PI/6, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	dubins_shortest_path(newDubins, victim_center[0].x, victim_center[0].y, M_PI/6, victim_center[1].x, victim_center[1].y, M_PI/6, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	dubins_shortest_path(newDubins, victim_center[1].x, victim_center[1].y, M_PI/6, victim_center[2].x, victim_center[2].y, M_PI/6, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	//dubins_shortest_path(newDubins, victim_center[2].x, victim_center[2].y, M_PI/6, victim_center[3].x, victim_center[3].y, M_PI/6, kmax);

	dubins_shortest_path(newDubins, victim_center[2].x, victim_center[2].y, M_PI/6, gateX, gateY, gateTh, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	/*dubins_shortest_path(newDubins, victim_center[3].x, victim_center[3].y, M_PI/6, victim_center[4].x, victim_center[4].y, M_PI/6, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	dubins_shortest_path(newDubins, victim_center[4].x, victim_center[4].y, M_PI/6, victim_center[5].x, victim_center[5].y, M_PI/6, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3

	dubins_shortest_path(newDubins, victim_center[5].x, victim_center[5].y, M_PI/6, gateX, gateY, gateTh, kmax);
	discretize_arc(newDubins.arc_1, s, npts, path);	// Arc 1
	discretize_arc(newDubins.arc_2, s, npts, path);	// Arc 2
	discretize_arc(newDubins.arc_3, s, npts, path);	// Arc 3*/

  }    

}

