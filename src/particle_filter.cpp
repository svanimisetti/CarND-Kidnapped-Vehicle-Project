/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/QR"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	// Set the number of particles.
	num_particles = 100;

	// Define a random number generator
	default_random_engine gen;
	
	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(0.0, std[0]);
	normal_distribution<double> dist_y(0.0, std[1]);
	normal_distribution<double> dist_theta(0.0, std[2]);
	
	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add Gaussian noise for GPS by sampling from above distribution.
	particles.clear();
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double dt, double std_pos[],
								double v, double thetad) {
	
	// Add measurements to each particle and add random Gaussian noise.
	// Adding noise you std::normal_distribution and std::default_random_engine
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/

	// Define a random number generator
	default_random_engine gen;
	
	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		
		// define local variables for ease of reading
		double x0, xf, y0, yf, theta0, thetaf;

		// Update based on motion model
		if(thetad<0.001) {
			particles[i].x += dist_x(gen) + v*dt*cos(particles[i].theta);
			particles[i].y += dist_y(gen) + v*dt*sin(particles[i].theta);
		} else {
			particles[i].x += dist_x(gen)
				+ (v/thetad)*(sin(particles[i].theta+thetad*dt)
							 -sin(particles[i].theta));
			particles[i].y += dist_y(gen)
				+ (v/thetad)*(cos(particles[i].theta)
							 -cos(particles[i].theta+thetad*dt));
		}
		particles[i].theta += dist_theta(gen) + thetad*dt;

	}
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted,
									 std::vector<LandmarkObs>& observations) {
	
	// Get predicted measurement that is closest to each observed
	// measurement and assign the observed measurement to this
	// particular landmark.

	double d_lm, min_d;
	int min_d_idx;
	for(int i=0; i<predicted.size(); i++) {
		min_d_idx = -1;
		min_d = 1.0e+6;
		for(int j=0; j<observations.size(); j++) {
			d_lm = dist(predicted[i].x, predicted[i].y,
						observations[j].x, observations[j].y);
			if(d_lm < min_d) {
				min_d = d_lm;
				min_d_idx = j;
			}
		}
		if(min_d_idx != -1 && min_d != 1.0e+6) {
			predicted[i].id = observations[min_d_idx].id;
			//std::cout << "DIST: " << min_d << ", ID: " << min_d_idx << std::endl;
			//std::cout << "P > " << predicted[i].id << "," << predicted[i].x << "," << predicted[i].y << std::endl;
			//std::cout << "O > " << observations[min_d_idx].id << "," << observations[min_d_idx].x << "," << observations[min_d_idx].y << std::endl;
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range,
								   double std_lm[], 
								   std::vector<LandmarkObs> obs_lm,
								   Map map_lm) {
	
	// Update particle weights using multi-variate Gaussian distribution.
	// 1. Observations are in VEHICLE'S coordinate system.
	// 2. Particles are in MAP's coordinate system.
	// 3. Transform using both rotation AND translation (no scaling)
	//    See http://planning.cs.uiuc.edu/node99.html for more info.
	// 4. PDF for multi-variate non-degenerative cases (since covariance matrix
	//    is positive definite), the following equation can be used:
	//    See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
	// 5. Transformation of covariance matrix between frames (see page 6 of following)
	//    http://ccar.colorado.edu/ASEN5070/lectures/Old_Lectures/ASEN_5070_2008_Lecture_27_Supplement.pdf

	// NOTES:
	// The covariance matrix of the measuremet is defined in vehicle frame.
	// For general cases, covariance matrix of measurement for each particle
	// has to be transformed from from the vehicle frame to world frame.
	// Therefore, the general case for computing the probability of
	// each observation for each particle is as follows:
	// rot_mat = [cos(theta) -sin(theta); sin(theta) cos(theta)];
	// cov_mat_t = rot_mat*cov_mat*rot_mat';
	// P(x,y) = exp[-(x_vec-mu_vec)*inv(cov_mat_t)*(x_vec-mu_vec)'] ...
	//        / (sqrt(det(2*pi*cov_mat_t)));
	// Special Case #1:
	// No cross-correlation
	//   => sig_xy=sig_yx=0 & sig_x~=sig_y
	// Even in this case, since transformation of covariance matrix from
	// vehicle to world frame will lead to cross-correlation terms, the
	// above formale will have to be used.
	// Special Case #2:
	// Equal variance with no cross-correlation
	//   => sig_xy=sig_yx=0 & sig_x==sig_y==sig
	// The covariance matrix does not change as the spread is a circle
	// after transformation to any frame is also a circle. Therefore,
	// the probability changes to following form:
	// P(x,y) = exp[-((x-mu_x)^2+(y-mu_y)^2)/(2*sig^2)]/(2*pi*sig^2)

	double p_weight = 1.0;
	double w_sum = 0.0;
	double cos_theta = 0.0, sin_theta = 0.0;
	Eigen::VectorXd ptx_vec(2);
	Eigen::VectorXd obs_vec(2);
	Eigen::VectorXd glb_vec(2);
	Eigen::MatrixXd cov_lm(2,2);
	Eigen::MatrixXd rot_L2G(2,2);
	std::vector<Map::single_landmark_s> lm_list = map_lm.landmark_list;
  
	vector<LandmarkObs> predicted_vec;
	vector<LandmarkObs> observations_vec;
	
	double d_range = 0.0;

	weights.clear();
	
	for(int i=0; i<num_particles; i++) {
		
		cos_theta = cos(particles[i].theta);
		sin_theta = sin(particles[i].theta);
		rot_L2G << cos_theta, -sin_theta, sin_theta, cos_theta;

		// Convert predicted observation to global w.r.t. particle's
		// position and associate with the closest landmark
		predicted_vec.clear();
		for(int j=0; j<obs_lm.size(); j++) {
			LandmarkObs pred_temp;
			ptx_vec << particles[i].x, particles[i].y;
			obs_vec << obs_lm[j].x, obs_lm[j].y;
			glb_vec = rot_L2G*obs_vec + ptx_vec;
			pred_temp.id = -1;
			pred_temp.x = glb_vec[0];
			pred_temp.y = glb_vec[1];
			predicted_vec.push_back(pred_temp);
		}

		// Compute actual observations from lidar data
		observations_vec.clear();
		for(int k=0; k<lm_list.size(); k++) {
			d_range = dist(particles[i].x, particles[i].y,
						   lm_list[k].x_f, lm_list[k].y_f);
			if(d_range <= sensor_range) {
				LandmarkObs obs_temp;
				obs_temp.id = lm_list[k].id_i;
				obs_temp.x = lm_list[k].x_f;
				obs_temp.y = lm_list[k].y_f;
				observations_vec.push_back(obs_temp);
			}
		}

		//std::cout << "Sizes: " << predicted_vec.size() << ", " << observations_vec.size() << std::endl;
		// Perform data associations
		dataAssociation(predicted_vec, observations_vec);
		//std::cout << "P_Vec: " << predicted_vec << std::endl;
		//std::cout << "O_Vec: " << observations_vec << std::endl;
		
		// Compute particle weights

		// Following is the general implementation of P(x,y) for each particle
		// Compute the rotation matrix and rotate cov_lm matrix to global
		p_weight = 1.0;
		double num, den;
		// For generalized implementation including cross-correlation
		//Eigen::VectorXd x_mu_vec(2);
		//cov_lm << std_lm[0]*std_lm[0], 0.0, 0.0, std_lm[1]*std_lm[1];
		//cov_lm = rot_L2G*cov_lm*rot_L2G.transpose();
		for(int j=0; j<predicted_vec.size(); j++) {

			for(int k=0; k<observations_vec.size(); k++) {
			
				if(predicted_vec[j].id != observations_vec[k].id) {
					continue;
				}
				
				/*For generalized implementation including cross-correlation
				x_mu_vec << (predicted_vec[j].x-observations_vec[k].y),
							(predicted_vec[j].y-observations_vec[k].y);
				double num = 0.5*x_mu_vec.transpose()*cov_lm.inverse()*x_mu_vec;
				double den = sqrt((2.0*EIGEN_PI*cov_lm).determinant());
				*/

				// Simple implementation assuming no cross-correlation
				num = pow((predicted_vec[j].x-observations_vec[k].x),2)
					  / (2.0*pow(std_lm[0],2))
					+ pow((predicted_vec[j].y-observations_vec[k].y),2)
					  / (2.0*pow(std_lm[1],2));
				double den = (2.0*M_PI*std_lm[0]*std_lm[1]);

				p_weight *= exp(-num/den);

			}
				
		}
		
		//std::cout << i << " : " << p_weight << std::endl;
		particles[i].weight = p_weight;
		weights.push_back(p_weight);
		w_sum += p_weight;

		// RMSE errors for state variables can be computed using
		// rmse_pos = sqrt((obs_x-meas_x)^2+(obs_y-meas_y)^2);
		// rmse_theta = sqrt((obs_theta-meas_theta)^2);

	}

	// Normalize the weights ensure summation to 1.0
	// Important when used in the resampling step with discrete_distribution
	for(int i=0; i<num_particles; i++) {
		weights[i] /= w_sum;
	}

}

void ParticleFilter::resample() {

	// Resample particles with probability proportional to their weight. 
	// Used std::discrete_distribution for implemenation
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Define a random number generator
	default_random_engine gen;
	
	// Create discrete distributions for x, y and theta
	discrete_distribution<int> dist(weights.begin(), weights.end());
	
	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add Gaussian noise for GPS by sampling from above distribution.
	std::vector<Particle> new_particles;
	new_particles.clear();
	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[dist(gen)]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle,
										 std::vector<int> associations,
										 std::vector<double> sense_x,
										 std::vector<double> sense_y) {
	
	// particle: Particle to assign listed association, and its (x,y) world coord
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
