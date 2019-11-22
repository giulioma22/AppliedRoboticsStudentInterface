#include "student_image_elab_interface.hpp"
#include "student_planning_interface.hpp"

#include <stdexcept>
#include <sstream>

#include <vector>
#include <atomic>
#include <unistd.h>

#include <experimental/filesystem>

namespace student {

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

    return true;
  }

  bool detect_green_victims(const cv::Mat& hsv_img, const double scale, std::vector<std::pair<int,Polygon>>& victim_list){
    
    cv::Mat green_mask_victims;
    //cv::inRange(hsv_img, cv::Scalar(50, 80, 34), cv::Scalar(75, 255, 255), green_mask_victims);
    cv::inRange(hsv_img, cv::Scalar(13, 68, 41), cv::Scalar(86, 255, 80), green_mask_victims);

    // Find green regions - VICTIMS
    std::vector<std::vector<cv::Point>> contours, contours_approx;
    std::vector<cv::Point> approx_curve;
    // Process green mask - VICTIMS
    cv::findContours(green_mask_victims, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (int i=0; i<contours.size(); ++i)
    {
      // Approximate polygon w/ fewer vertices if not precise
      approxPolyDP(contours[i], approx_curve, 1, true);

      Polygon scaled_contour;
      for (const auto& pt: approx_curve) {
        scaled_contour.emplace_back(pt.x/scale, pt.y/scale);
      }
      // Add victim to list
      victim_list.push_back({i+1, scaled_contour});
    }
    
    return true;
  }

  bool detect_green_gate(const cv::Mat& hsv_img, const double scale, Polygon& gate){

    cv::Mat green_mask_gate;    
    //cv::inRange(hsv_img, cv::Scalar(50, 80, 34), cv::Scalar(75, 255, 255), green_mask_gate);
    cv::inRange(hsv_img, cv::Scalar(13, 68, 41), cv::Scalar(86, 255, 80), green_mask_gate);
    
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

  bool detect_blue_robot(const cv::Mat& hsv_img, const double scale, Polygon& triangle, double& x, double& y, double& theta){

    cv::Mat blue_mask;    
    //cv::inRange(hsv_img, cv::Scalar(92, 80, 50), cv::Scalar(145, 255, 255), blue_mask);
    cv::inRange(hsv_img, cv::Scalar(96, 92, 55), cv::Scalar(145, 255, 255), blue_mask);

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
  

  }    

}

