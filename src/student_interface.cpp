#include "student_image_elab_interface.hpp"
#include "student_planning_interface.hpp"

#include <stdexcept>
#include <sstream>

#include <experimental/filesystem>

namespace student {

 void loadImage(cv::Mat& img_out, const std::string& config_folder){  
   throw std::logic_error( "STUDENT FUNCTION NOT IMPLEMENTED" );
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

  bool extrinsicCalib(const cv::Mat& img_in, std::vector<cv::Point3f> object_points, const cv::Mat& camera_matrix, cv::Mat& rvec, cv::Mat& tvec, const std::string& config_folder){

    std::string extrinsic_path = config_folder + "extrinsicCalib.csv";
    std::vector<cv::Point2f> imagePoints;
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

    bool result = cv::solvePnP(object_points, imagePoints, camera_matrix, {}, rvec, tvec);

    return result;

  }

  void imageUndistort(const cv::Mat& img_in, cv::Mat& img_out, 
          const cv::Mat& cam_matrix, const cv::Mat& dist_coeffs, const std::string& config_folder){

    cv::undistort(img_in, img_out, cam_matrix, dist_coeffs);

  }
 
  void findPlaneTransform(const cv::Mat& cam_matrix, const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point3f>& object_points_plane, const std::vector<cv::Point2f>& dest_image_points_plane, cv::Mat& plane_transf, const std::string& config_folder){

    cv::Mat image_points;
    cv::projectPoints(object_points_plane, rvec, tvec, cam_matrix, cv::Mat(), image_points);
    plane_transf = cv::getPerspectiveTransform(image_points, dest_image_points_plane);

  }

  void unwarp(const cv::Mat& img_in, cv::Mat& img_out, const cv::Mat& transf, const std::string& config_folder){
    cv::warpPerspective(img_in, img_out, transf, img_in.size());
  }

  bool processMap(const cv::Mat& img_in, const double scale, std::vector<Polygon>& obstacle_list, std::vector<std::pair<int,Polygon>>& victim_list, Polygon& gate, const std::string& config_folder){
    throw std::logic_error( "STUDENT FUNCTION NOT IMPLEMENTED 3" );   
  }

  bool findRobot(const cv::Mat& img_in, const double scale, Polygon& triangle, double& x, double& y, double& theta, const std::string& config_folder){
    throw std::logic_error( "STUDENT FUNCTION NOT IMPLEMENTED 4" );    
  }

  bool planPath(const Polygon& borders, const std::vector<Polygon>& obstacle_list, const std::vector<std::pair<int,Polygon>>& victim_list, const Polygon& gate, const float x, const float y, const float theta, Path& path){
    throw std::logic_error( "STUDENT FUNCTION NOT IMPLEMENTED 5" );     
  }


}

