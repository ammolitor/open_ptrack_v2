/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * head_based_subcluster.hpp
 * Created on: Nov 30, 2012
 * Author: Matteo Munaro
 */

#ifndef OPEN_PTRACK_PERSON_CLUSTERING_HEAD_BASED_SUBCLUSTER_HPP_
#define OPEN_PTRACK_PERSON_CLUSTERING_HEAD_BASED_SUBCLUSTER_HPP_

#include <open_ptrack/person_clustering/head_based_subclustering.h>
#include <iostream>

template <typename PointT>
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::HeadBasedSubclustering ()
{
  // set default values for optional parameters:
  vertical_ = false;
  head_centroid_ = true;
  min_height_ = 1.3;
  max_height_ = 2.3;
  min_points_ = 30;
  max_points_ = 5000;
  heads_minimum_distance_ = 0.3;

  // set flag values for mandatory parameters:
  sqrt_ground_coeffs_ = std::numeric_limits<float>::quiet_NaN();
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setInputCloud (PointCloudPtr& cloud)
{
  cloud_ = cloud;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setGround (Eigen::VectorXf& ground_coeffs)
{
  ground_coeffs_ = ground_coeffs;
  sqrt_ground_coeffs_ = (ground_coeffs - Eigen::Vector4f(0.0f, 0.0f, 0.0f, ground_coeffs(3))).norm();
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setInitialClusters (std::vector<pcl::PointIndices>& cluster_indices)
{
  cluster_indices_ = cluster_indices;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setSensorPortraitOrientation (bool vertical)
{
  vertical_ = vertical;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setHeightLimits (float min_height, float max_height)
{
  min_height_ = min_height;
  max_height_ = max_height;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setDimensionLimits (int min_points, int max_points)
{
  min_points_ = min_points;
  max_points_ = max_points;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setMinimumDistanceBetweenHeads (float heads_minimum_distance)
{
  heads_minimum_distance_= heads_minimum_distance;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::setHeadCentroid (bool head_centroid)
{
  head_centroid_ = head_centroid;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::getHeightLimits (float& min_height, float& max_height)
{
  min_height = min_height_;
  max_height = max_height_;
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::getDimensionLimits (int& min_points, int& max_points)
{
  min_points = min_points_;
  max_points = max_points_;
}

template <typename PointT> float
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::getMinimumDistanceBetweenHeads ()
{
  return (heads_minimum_distance_);
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::mergeClustersCloseInFloorCoordinates (std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& input_clusters,
    std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& output_clusters)
{
  float min_distance_between_cluster_centers = 0.4;                   // meters
  float normalize_factor = std::pow(sqrt_ground_coeffs_, 2);          // sqrt_ground_coeffs ^ 2 (precomputed for speed)
  Eigen::Vector3f head_ground_coeffs = ground_coeffs_.head(3);        // ground plane normal (precomputed for speed)
  std::vector <std::vector<int> > connected_clusters;
  connected_clusters.resize(input_clusters.size());
  std::vector<bool> used_clusters;          // 0 in correspondence of clusters remained to process, 1 for already used clusters
  used_clusters.resize(input_clusters.size());
  for(unsigned int i = 0; i < input_clusters.size(); i++)             // for every cluster
  {
    Eigen::Vector3f theoretical_center = input_clusters[i].getTCenter();
    float t = theoretical_center.dot(head_ground_coeffs) / normalize_factor;    // height from the ground
    Eigen::Vector3f current_cluster_center_projection = theoretical_center - head_ground_coeffs * t;    // projection of the point on the groundplane
    for(unsigned int j = i+1; j < input_clusters.size(); j++)         // for every remaining cluster
    {
      theoretical_center = input_clusters[j].getTCenter();
      float t = theoretical_center.dot(head_ground_coeffs) / normalize_factor;    // height from the ground
      Eigen::Vector3f new_cluster_center_projection = theoretical_center - head_ground_coeffs * t;      // projection of the point on the groundplane
      if (((new_cluster_center_projection - current_cluster_center_projection).norm()) < min_distance_between_cluster_centers)
      {
        connected_clusters[i].push_back(j);
      }
    }
  }

  for(unsigned int i = 0; i < connected_clusters.size(); i++)   // for every cluster
  {
    if (!used_clusters[i])                                      // if this cluster has not been used yet
    {
      used_clusters[i] = true;
      if (connected_clusters[i].empty())                        // no other clusters to merge
      {
        output_clusters.push_back(input_clusters[i]);
      }
      else
      {
        // Copy cluster points into new cluster:
        pcl::PointIndices point_indices;
        point_indices = input_clusters[i].getIndices();
        for(unsigned int j = 0; j < connected_clusters[i].size(); j++)
        {
          if (!used_clusters[connected_clusters[i][j]])         // if this cluster has not been used yet
          {
            used_clusters[connected_clusters[i][j]] = true;
            for(std::vector<int>::const_iterator points_iterator = input_clusters[connected_clusters[i][j]].getIndices().indices.begin();
                points_iterator != input_clusters[connected_clusters[i][j]].getIndices().indices.end(); points_iterator++)
            {
              point_indices.indices.push_back(*points_iterator);
            }
          }
        }
        open_ptrack::person_clustering::PersonCluster<PointT> cluster(cloud_, point_indices, ground_coeffs_, sqrt_ground_coeffs_, head_centroid_, vertical_);
        output_clusters.push_back(cluster);
      }
    }
  }
    }

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::createSubClusters (open_ptrack::person_clustering::PersonCluster<PointT>& cluster, int maxima_number,
    std::vector<int>& maxima_cloud_indices, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& subclusters)
{
  // create new clusters from the current cluster and put corresponding indices into sub_clusters_indices:
  float normalize_factor = std::pow(sqrt_ground_coeffs_, 2);          // sqrt_ground_coeffs ^ 2 (precomputed for speed)
  Eigen::Vector3f head_ground_coeffs = ground_coeffs_.head(3);        // ground plane normal (precomputed for speed)
  Eigen::Matrix3Xf maxima_projected(3,maxima_number);                 // matrix containing the projection of maxima onto the ground plane
  Eigen::VectorXi subclusters_number_of_points(maxima_number);        // subclusters number of points
  std::vector <std::vector <int> > sub_clusters_indices;              // vector of vectors with the cluster indices for every maximum
  sub_clusters_indices.resize(maxima_number);                         // resize to number of maxima

  // Project maxima on the ground plane:
  for(int i = 0; i < maxima_number; i++)                              // for every maximum
  {
    PointT* current_point = &cloud_->points[maxima_cloud_indices[i]]; // current maximum point cloud point
    Eigen::Vector3f p_current_eigen(current_point->x, current_point->y, current_point->z);  // conversion to eigen
    float t = p_current_eigen.dot(head_ground_coeffs) / normalize_factor;       // height from the ground
    maxima_projected.col(i).matrix () = p_current_eigen - head_ground_coeffs * t;         // projection of the point on the groundplane
    subclusters_number_of_points(i) = 0;                              // intialize number of points
  }

  // Associate cluster points to one of the maximum:
  for(std::vector<int>::const_iterator points_iterator = cluster.getIndices().indices.begin(); points_iterator != cluster.getIndices().indices.end(); points_iterator++)
  {
    PointT* current_point = &cloud_->points[*points_iterator];        // current point cloud point
    Eigen::Vector3f p_current_eigen(current_point->x, current_point->y, current_point->z);  // conversion to eigen
    float t = p_current_eigen.dot(head_ground_coeffs) / normalize_factor;       // height from the ground
    p_current_eigen = p_current_eigen - head_ground_coeffs * t;       // projection of the point on the groundplane

    int i = 0;
    bool correspondence_detected = false;
    while ((!correspondence_detected) && (i < maxima_number))
    {
      if (((p_current_eigen - maxima_projected.col(i)).norm()) < heads_minimum_distance_)
      {
        correspondence_detected = true;
        sub_clusters_indices[i].push_back(*points_iterator);
        subclusters_number_of_points(i)++;
      }
      else
        i++;
    }
  }

  // Create a subcluster if the number of points associated to a maximum is over a threshold:
  for(int i = 0; i < maxima_number; i++)                              // for every maximum
  {
    if (subclusters_number_of_points(i) > min_points_)
    {
      pcl::PointIndices point_indices;
      point_indices.indices = sub_clusters_indices[i];                // indices associated to the i-th maximum

      open_ptrack::person_clustering::PersonCluster<PointT> cluster(cloud_, point_indices, ground_coeffs_, sqrt_ground_coeffs_, head_centroid_, vertical_);
      subclusters.push_back(cluster);
      //std::cout << "Cluster number of points: " << subclusters_number_of_points(i) << std::endl;
    }
  }
}

template <typename PointT> void
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::subcluster (std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters)
{
  // Check if all mandatory variables have been set:
  if (sqrt_ground_coeffs_ != sqrt_ground_coeffs_)
  {
    std::cout << "[HeadBasedSubclustering::subcluster] Floor parameters have not been set or they are not valid!" << std::endl;
    PCL_ERROR ("[pcl::people::open_ptrack::person_clustering::HeadBasedSubclustering::subcluster] Floor parameters have not been set or they are not valid!\n");
    return;
  }
  if (cluster_indices_.size() == 0)
  {
    std::cout << "[HeadBasedSubclustering::subcluster] Cluster indices have not been set!" << std::endl;
    PCL_ERROR ("[pcl::people::open_ptrack::person_clustering::HeadBasedSubclustering::subcluster] Cluster indices have not been set!\n");
    return;
  }
  if (cloud_ == NULL)
  {
    std::cout << "[HeadBasedSubclustering::subcluster] Input cloud has not been set!" << std::endl;
    PCL_ERROR ("[pcl::people::open_ptrack::person_clustering::HeadBasedSubclustering::subcluster] Input cloud has not been set!\n");
    return;
  }

  // Person clusters creation from clusters indices:
  std::cout << "[HeadBasedSubclustering::subcluster] clusters.size: " << clusters.size() << std::endl;
  if (clusters.size() == 0){
    for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices_.begin(); it != cluster_indices_.end(); ++it)
    {
      open_ptrack::person_clustering::PersonCluster<PointT> cluster(cloud_, *it, ground_coeffs_, sqrt_ground_coeffs_, head_centroid_, vertical_);  // PersonCluster creation
      clusters.push_back(cluster);
    }
  }

  // Remove clusters with too high height from the ground plane:
  std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > new_clusters;
  for(unsigned int i = 0; i < clusters.size(); i++)   // for every cluster
  {
    if (clusters[i].getHeight() <= max_height_)
      new_clusters.push_back(clusters[i]);
  }
  clusters = new_clusters;
  new_clusters.clear();
  std::cout << "[HeadBasedSubclustering::subcluster] max_height completed" << std::endl;

  // Merge clusters close in floor coordinates:
  mergeClustersCloseInFloorCoordinates(clusters, new_clusters); // failed here in dark
  clusters = new_clusters;
  std::cout << "[HeadBasedSubclustering::subcluster] mergeClustersCloseInFloorCoordinates completed" << std::endl;

  std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > subclusters;
  int cluster_min_points_sub = int(float(min_points_) * 1.5);
  //  int cluster_max_points_sub = max_points_;

  // create HeightMap2D object:
  open_ptrack::person_clustering::HeightMap2D<PointT> height_map_obj;
  height_map_obj.setGround(ground_coeffs_);
  std::cout << "[HeadBasedSubclustering::subcluster] height_map_obj.setGround completed" << std::endl;
  std::cout << "[HeadBasedSubclustering::subcluster] cloud check: " << cloud_.height << std::endl;
  height_map_obj.setInputCloud(cloud_);
  std::cout << "[HeadBasedSubclustering::subcluster] height_map_obj.setInputCloud completed" << std::endl;
  height_map_obj.setSensorPortraitOrientation(vertical_);
  std::cout << "[HeadBasedSubclustering::subcluster] height_map_obj.setSensorPortraitOrientation completed" << std::endl;
  height_map_obj.setMinimumDistanceBetweenMaxima(heads_minimum_distance_);
  std::cout << "[HeadBasedSubclustering::subcluster] height_map_obj.setMinimumDistanceBetweenMaxima completed" << std::endl;
  for(typename std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)   // for every cluster
  {
    float height = it->getHeight();
    int number_of_points = it->getNumberPoints();
    std::cout << "[HeadBasedSubclustering::subcluster] height: " << height << " number_of_points: " << number_of_points << std::endl;
    
    if(height > min_height_ && height < max_height_)
    {
      std::cout << "[HeadBasedSubclustering::subcluster] height > min_height_ && height < max_height_" << std::endl;
      if (number_of_points > cluster_min_points_sub) //  && number_of_points < cluster_max_points_sub)
      {
        // Compute height map associated to the current cluster and its local maxima (heads):
        std::cout << "[HeadBasedSubclustering::subcluster] number_of_points > cluster_min_points_sub" << std::endl;
        height_map_obj.compute(*it);
        std::cout << "[HeadBasedSubclustering::subcluster] height_map_obj.compute(*it) finsihed" << std::endl;
        if (height_map_obj.getMaximaNumberAfterFiltering() > 1)        // if more than one maximum
        {
          // create new clusters from the current cluster and put corresponding indices into sub_clusters_indices:
          createSubClusters(*it, height_map_obj.getMaximaNumberAfterFiltering(), height_map_obj.getMaximaCloudIndicesFiltered(), subclusters);
          std::cout << "[HeadBasedSubclustering::subcluster] createSubClusters finsihed" << std::endl;
        }
        else
        {  // Only one maximum --> copy original cluster:
          subclusters.push_back(*it);
          std::cout << "[HeadBasedSubclustering::subcluster] Only one maximum --> copy original cluster:" << std::endl;
        }
      }
      else
      {
        // Cluster properties not good for sub-clustering --> copy original cluster:
        subclusters.push_back(*it);
        std::cout << "[HeadBasedSubclustering::subcluster]  Cluster properties not good for sub-clustering --> copy original cluster" << std::endl;
      }
    }
  }
  clusters = subclusters;    // substitute clusters with subclusters
  std::cout << "[HeadBasedSubclustering::subcluster] finished!" << std::endl;
}

template <typename PointT>
open_ptrack::person_clustering::HeadBasedSubclustering<PointT>::~HeadBasedSubclustering ()
{
  // TODO Auto-generated destructor stub
}
#endif /* OPEN_PTRACK_PERSON_CLUSTERING_HEAD_BASED_SUBCLUSTER_HPP_ */
