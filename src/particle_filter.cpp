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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// create normal distributions for x, y, and theta

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	default_random_engine gen;

	// Set number of particles
	num_particles = 100;

	// generate the particles
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
	    particle.id = i;
	    particle.x = dist_x(gen);
	    particle.y = dist_y(gen);
	    particle.theta = dist_theta(gen);
	    particle.weight = 1.0;

	    particles.push_back(particle);
	  }

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	std::default_random_engine gen;

	// generate random Gaussian noise
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++){

		double theta = particles[i].theta;

	    // add measurements to each particle
	    if( fabs(yaw_rate) < 0.0001){  // constant velocity
	    	particles[i].x += velocity * delta_t * cos(theta);
	    	particles[i].y += velocity * delta_t * sin(theta);

	    } else{
	    	particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate*delta_t ) - sin(theta) );
	    	particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate*delta_t ) );
	    	particles[i].theta += yaw_rate * delta_t;
	    }

	    // predicted particles with added sensor noise
	    particles[i].x += dist_x(gen);
	    particles[i].y += dist_y(gen);
	    particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	unsigned int nObservations = observations.size();
	unsigned int nPredictions = predicted.size();

	for (unsigned int i = 0; i < nObservations; i++) {
		double minD = numeric_limits<float>::max();

		int mapId = -1;

		for (unsigned j = 0; j < nPredictions; j++ ) {
	      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
	      if( minD > distance){
	        minD = distance;
	        mapId = predicted[j].id;
	      }
	    }
		observations[i].id = mapId;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for(int i = 0; i < num_particles; i++){
		particles[i].weight = 1.0;

		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;

	    // collect valid landmarks
	    vector<LandmarkObs> predictions;
	    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
	    	float lmx_f = map_landmarks.landmark_list[j].x_f;
	    	float lmy_f = map_landmarks.landmark_list[j].y_f;
	    	int lmid_i = map_landmarks.landmark_list[j].id_i;

	      double distance = dist(px, py, lmx_f, lmy_f);
	      if( distance < sensor_range){ // if the landmark is within the sensor range, save it to predictions
	        predictions.push_back(LandmarkObs{lmid_i, lmx_f, lmy_f});
	      }
	    }

	    // convert observations coordinates from vehicle to map
	    vector<LandmarkObs> observations_map;
	    double cos_theta = cos(ptheta);
	    double sin_theta = sin(ptheta);

	    for(unsigned int j = 0; j < observations.size(); j++){
	      LandmarkObs tmp;
	      double obsx = observations[j].x;
	      double obsy = observations[j].y;

	      tmp.x = obsx * cos_theta - obsy * sin_theta + px;
	      tmp.y = obsx * sin_theta + obsy * cos_theta + py;
	      observations_map.push_back(tmp);
	    }

	    // find landmark index for each observation
	    dataAssociation(predictions, observations_map);

	    // compute the particle's weight:
	    for(unsigned int j = 0; j < observations_map.size(); j++){
	    	double obs_mx = observations_map[j].x;
	    	double obs_my = observations_map[j].y;
	    	int obs_mid = observations_map[j].id;

	      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_mid-1);
	      double x_term = pow(obs_mx - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
	      double y_term = pow(obs_my - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
	      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	      particles[i].weight *=  w;
	    }

	    weights.push_back(particles[i].weight);

	  }

}

void ParticleFilter::resample() {
	// generate distribution according to weights

	default_random_engine gen;

	discrete_distribution<> dist(weights.begin(), weights.end());

	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);

	for(int i=0; i<num_particles; i++){
		int idx = dist(gen);
		resampled_particles[i] = particles[idx];
	}

	particles = resampled_particles;
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
