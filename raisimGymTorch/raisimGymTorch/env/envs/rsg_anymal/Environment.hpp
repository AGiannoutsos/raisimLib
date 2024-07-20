//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include <chrono>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    // set rng
    timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    unif = std::uniform_real_distribution<double>(-4, 4);
    startingTime = std::time(0);


    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    Joint_angles.setZero(12);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();

    // change randomly the target speed
    // radomTargetVelocity = unif(rng);
    // radomTargetVelocity = std::sin((1.0*(double)std::time(0) / 60.0));
    // double timePeriod = (2.0*(double)std::time(0) / 60.0);
    // radomTargetVelocity = std::sin(2*std::sin(timePeriod));


    // int minutes = 5;
    // if ( ((int)(((double)std::time(0) - startingTime)) % (60*minutes)) < (30*minutes) ) {
    //   radomTargetVelocity = 0.5;
    // }
    // else {
    //   radomTargetVelocity = -0.5;
    // }

    
    radomTargetVelocity = 0.5;
  }

  double getPositiveTargetVelocity(double velocity) {
    return std::min(radomTargetVelocity, velocity);
  }

  double getNegativeTargetVelocity(double velocity) {
    return -std::max(radomTargetVelocity, velocity);
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    // penalize hight torque
    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());

    // horizontal speed
    // rewards_.record("forwardVel", -std::max(-1.0, bodyLinearVel_[0]));
    if (radomTargetVelocity > 0) {
      rewards_.record("forwardVel", getPositiveTargetVelocity(bodyLinearVel_[0]));
    } else {
      rewards_.record("forwardVel", getNegativeTargetVelocity(bodyLinearVel_[0]));
    }

    // calculate foot clearance
    raisim::Vec<3> LF_FOOT_DUMMY_JOINT_joint_position;
    raisim::Vec<3> RF_FOOT_DUMMY_JOINT_joint_position;
    raisim::Vec<3> LH_FOOT_DUMMY_JOINT_joint_position;
    raisim::Vec<3> RH_FOOT_DUMMY_JOINT_joint_position;
    anymal_->getFramePosition("LF_FOOT_DUMMY_JOINT", LF_FOOT_DUMMY_JOINT_joint_position);
    anymal_->getFramePosition("RF_FOOT_DUMMY_JOINT", RF_FOOT_DUMMY_JOINT_joint_position);
    anymal_->getFramePosition("LH_FOOT_DUMMY_JOINT", LH_FOOT_DUMMY_JOINT_joint_position);
    anymal_->getFramePosition("RH_FOOT_DUMMY_JOINT", RH_FOOT_DUMMY_JOINT_joint_position);
    auto foot_clearance = (
      LF_FOOT_DUMMY_JOINT_joint_position[2] + 
      RF_FOOT_DUMMY_JOINT_joint_position[2] + 
      LH_FOOT_DUMMY_JOINT_joint_position[2] + 
      RH_FOOT_DUMMY_JOINT_joint_position[2]
    );
    auto mean_foot_clearance = foot_clearance / 4.0; 
    rewards_.record("footClearance", std::min(2.0, mean_foot_clearance));

    // remove jumping behaviour by penalizing the body height movement
    rewards_.record("bodyHeight", std::abs(gc_[2] - 0.5));

    // penalize yz valocity movement
    rewards_.record("yVel", std::abs(bodyLinearVel_[1] - 0.0));
    rewards_.record("zVel", std::abs(bodyLinearVel_[2] - 0.0));

    // penalize angulat speed
    rewards_.record("angVelocity",  std::abs(bodyAngularVel_[2] - 0.0));

    
    

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    // raisim::Vec<3> LF_FOOT_world_position;
    // raisim::Vec<3> LF_HAA_joint_position;
    // raisim::Vec<3> LF_HFE_joint_position;
    // raisim::Vec<3> LF_KFE_joint_position;
    // raisim::Vec<3> LF_ADAPTER_TO_FOOT_joint_position;
    // raisim::Vec<3> LF_HAA_link_position;
    // raisim::Vec<3> LF_HFE_link_position;
    // raisim::Vec<3> LF_KFE_link_position;
    // raisim::Vec<3> LF_ADAPTER_TO_FOOT_link_position;
    // anymal_->getPosition(anymal_->getBodyIdx("LF_KFE"), LF_FOOT_world_position);
    // anymal_->getFramePosition("LF_HAA", LF_HAA_joint_position);
    // anymal_->getFramePosition("LF_HFE", LF_HFE_joint_position);
    // anymal_->getFramePosition("LF_KFE", LF_KFE_joint_position);
    // anymal_->getFramePosition("LF_ADAPTER_TO_FOOT", LF_ADAPTER_TO_FOOT_joint_position);
    // anymal_->getFramePosition(anymal_->getFrameByLinkName("LF_HIP"), LF_HAA_link_position);
    // anymal_->getFramePosition(anymal_->getFrameByLinkName("LF_THIGH"), LF_HFE_link_position);
    // anymal_->getFramePosition(anymal_->getFrameByLinkName("LF_SHANK"), LF_KFE_link_position);
    // anymal_->getFramePosition(anymal_->getFrameByLinkName("LF_ADAPTER"), LF_ADAPTER_TO_FOOT_link_position);
    
    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12); /// joint velocity
  }

  void getPosition(Eigen::Ref<EigenVec> po) {
    // get position on world frame
    raisim::Vec<3> Robot_world_position;
    anymal_->getPosition(anymal_->getBodyIdx("base"), Robot_world_position);

    // convert it to Eigen vec
    Eigen::Vector3d Robot_world_position_eigen;
    Robot_world_position_eigen << Robot_world_position[0], Robot_world_position[1], Robot_world_position[2];
    /// convert it to float
    po = Robot_world_position_eigen.cast<float>();
  }

  void getOrientation(Eigen::Ref<EigenVec> ori) {
    // get joint positions on their reference frame
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    // get euler angles
    double* array_quat = new double[4]; 
    double* euler_rot = new double[3]; 
    array_quat[0] = quat[0]; array_quat[1] = quat[1]; array_quat[2] = quat[2]; array_quat[3] = quat[3];
    raisim::quatToEulerVec(array_quat, euler_rot);

    // convert it to Eigen vec
    Eigen::Vector3d Robot_orientation;
    Robot_orientation << euler_rot[0], euler_rot[1], euler_rot[2];
    /// convert it to float
    ori = Robot_orientation.cast<float>();
    delete array_quat, euler_rot;
  }

  void getJointAngles(Eigen::Ref<EigenVec> ang) {
    // get joint angles on their reference frame
    anymal_->getState(gc_, gv_);

    // convert it to Eigen vec
    // Eigen::VectorXd Joint_angles;
    Joint_angles << gc_.tail(12);
    /// convert it to float
    ang = Joint_angles.cast<float>();
  }

  void getTargetVelocity(Eigen::Ref<EigenVec> tVel) {
     Eigen::VectorXd target_velocity;
     target_velocity.setZero(1);
     target_velocity << radomTargetVelocity;
    /// convert it to float
    tVel = target_velocity.cast<float>();
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  // RNG
  std::mt19937_64 rng;
  uint64_t timeSeed;
  std::seed_seq ss;
  std::uniform_real_distribution<double> unif;
  double radomTargetVelocity;
  int startingTime;

  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, Joint_angles;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

