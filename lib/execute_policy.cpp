//
// Created by yue on 5/23/18.
//
#include <boost/bind.hpp>
// ROS stuff
#include <ros/ros.h>
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Scalar.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <actionlib/client/simple_action_client.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <vicon_object_client.h>

#include <geometry_msgs/PoseStamped.h>
#include <moveit/robot_state/robot_state.h>

#include <geometry_msgs/TransformStamped.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <vicon_object_server_msgs/TrajGetWaypoint.h>

#include <iostream>

#include "my_utils.h"
#include "execute_policy.h"
#include "lab_domain.h"
#include "solver.h"
#include "partial_solver.h"

FetchMoveBaseDemo::FetchMoveBaseDemo():
        move_base_client_("move_base", true), tf_buffer_(), local_nh_("~"),
        tf_listener_(tf_buffer_) {

    //=======================================================================
    // Get parameters
    //=======================================================================
    local_nh_.param<std::string>("target_object_name", target_object_name_, "demo_cube");
    local_nh_.param<std::string>("target_object_grasping_traj_name", target_object_grasping_traj_name_,
                                 "Demo Cube Top Grasp");
    local_nh_.param<std::string>("table_name", table_name_,
                                 "cafe_table");
    local_nh_.param<std::string>("test_file_path", test_file_path_,
                                 "/home/yuew/ros/indigo/dev/src/boolean_pomdp_synthesizer/icra19_results/test_instances/lab_test");
    local_nh_.getParam("obstacle_names", obstacle_names_);
    local_nh_.param<double>("policy_stay_th", policy_stay_th_, 1.0);
    local_nh_.param<int>("horizon_bound", horizon_bound_, 10);

    ROS_INFO("Fetch frame: %s", fetch_frame_.c_str());
    ROS_INFO("Target object name: %s", target_object_name_.c_str());
    ROS_INFO("Target object Grasping Trajectory name: %s", target_object_grasping_traj_name_.c_str());
    ROS_INFO("Table name: %s", table_name_.c_str());
    ROS_INFO("Test file path: %s", test_file_path_.c_str());
    
    for (int i = 0; i < obstacle_names_.size(); ++i)  {
        ROS_INFO("Obstacle name: %s", obstacle_names_[i].c_str());
    }

    //wait for the action server to come up
    while(!move_base_client_.waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the move_base action server to come up");
    }
    ROS_INFO("Move_base action server is running");

    //=======================================================================
    // Action lib
    //=======================================================================
    std::string action_name = "gripper_controller/gripper_action";
    gripper_action_client_.reset(new gripper_action_client_t(action_name, true));
    if (!gripper_action_client_->waitForServer(ros::Duration(2.0)))
        ROS_ERROR("%s may not be connected.", action_name.c_str());

    replan_num_ = 0;

    policy_ = nullptr;

    planning_scene_diff_publisher_ =
            node_handler_.advertise<moveit_msgs::PlanningScene>("planning_scene", 1000);
}

std::vector<double> FetchMoveBaseDemo::getCurrentBasePose() {
    // listen to tf
    while (node_handler_.ok()) {
        try{
            base_to_map_tf_ = tf_buffer_.lookupTransform("map", "base_link",
                                                         ros::Time(0));
            double x = base_to_map_tf_.transform.translation.x;
            double y = base_to_map_tf_.transform.translation.y;
            double yaw = tf2::getYaw(base_to_map_tf_.transform.rotation);
            std::vector<double> ret(3);
            ret[0] = x;
            ret[1] = y;
            ret[2] = yaw;
            return ret;
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }
}

std::vector<double> FetchMoveBaseDemo::getCurrentObjectPose(const std::string &object_name) {
	// listen to tf
    while (node_handler_.ok()) {
        try{
            geometry_msgs::TransformStamped object_to_base_tf 
				= tf_buffer_.lookupTransform("map", "vicon_object_server/" + object_name, ros::Time(0));
            double x = object_to_base_tf.transform.translation.x;
            double y = object_to_base_tf.transform.translation.y;
            std::vector<double> ret(2);
            ret[0] = x;
            ret[1] = y;
            return ret;
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }
}

bool FetchMoveBaseDemo::observe(std::string direct) {
	std::vector<double> base_pose = getCurrentBasePose();
    double observe_loc_x;
    double observe_loc_y;
    if (direct == "forward") {
        observe_loc_x = base_pose[0] + forword_dist_one_step_;
        observe_loc_y = base_pose[1];
    } else if (direct == "down") {
        observe_loc_x = base_pose[0];
        observe_loc_y = base_pose[1] - forword_dist_one_step_;
    } else {
        ROS_INFO("Unknown direction to observe");
        return true;
    }

    for (int i = 0; i < obstacle_names_.size(); ++i) {
		std::vector<double> obstacle_pose = getCurrentObjectPose(obstacle_names_[i]);
        double obs_x = obstacle_pose[0];
        double obs_y = obstacle_pose[1];
        double dist = (observe_loc_x - obs_x) * (observe_loc_x - obs_x)
                + (observe_loc_y - obs_y) * (observe_loc_y - obs_y);
        if (dist < 0.5 * 0.5) {
            double rand_point = MyUtils::uniform01();
            if (rand_point > LabDomain::p_fn) {
                ROS_INFO("Observe obstacle: %s", obstacle_names_[i].c_str());
                return true;
            } else {
                ROS_INFO("No obstacle observed (False Negative)");
                return false;
            }

        }
    }
    double rand_point = MyUtils::uniform01();
    if (rand_point > LabDomain::p_fp) {
        ROS_INFO("No obstacle observed");
        return false;
    } else {
        ROS_INFO("Observe obstacle (False Positive)");
        return true;
    }

}

void FetchMoveBaseDemo::addBaseBox()
{
    moveit_msgs::PlanningScene planning_scene_collision;
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = "/base_link";
    collision_object.id = "keep_out";
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.2;
    primitive.dimensions[1] = 0.5;
    primitive.dimensions[2] = 0.05;
    geometry_msgs::Pose box_pose;
    box_pose.orientation.w = 1;
    box_pose.position.x = 0.15;
    box_pose.position.y = 0;
    box_pose.position.z = 0.375;
    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    planning_scene_collision.world.collision_objects.push_back(collision_object);
    planning_scene_collision.is_diff = true;
    while (planning_scene_diff_publisher_.getNumSubscribers() < 1)
    {
        ros::WallDuration sleep_t(0.5);
        sleep_t.sleep();
    }
    // Publish the added object
    planning_scene_diff_publisher_.publish(planning_scene_collision);
}

void FetchMoveBaseDemo::addUncertainBoxes() {

    moveit_msgs::PlanningScene planning_scene;

    for (int i = 1; i < LabDomain::x_size - 1; ++i) {
        for (int j = 0; j < LabDomain::y_size; ++j) {
            int idx = (i - 1) * LabDomain::y_size + j;

            auto xy = uncertain_boxes[idx];
            double x = xy.first;
            double y = xy.second;

            moveit_msgs::CollisionObject collision_object;
            collision_object.header.frame_id = "/map";
            collision_object.id = "uncertain_box_" + std::to_string(i) + "_" + std::to_string(j);
            shape_msgs::SolidPrimitive primitive;
            primitive.type = primitive.BOX;
            primitive.dimensions.resize(3);
            primitive.dimensions[0] = forword_dist_one_step_;
            primitive.dimensions[1] = forword_dist_one_step_;
            primitive.dimensions[2] = 1.0;
            geometry_msgs::Pose box_pose;
            box_pose.orientation.w = 1;
            box_pose.position.x = x;
            box_pose.position.y = y;
            box_pose.position.z = 0.5;
            collision_object.primitives.push_back(primitive);
            collision_object.primitive_poses.push_back(box_pose);
            collision_object.operation = collision_object.ADD;
            planning_scene.world.collision_objects.push_back(collision_object);

            moveit_msgs::ObjectColor color;
            if (current_probs[idx] > LabDomain::delta) {
                color.color.r = 1;
                color.color.g = 1;
                color.color.b = 0;
                color.color.a = max(min(current_probs[idx] / init_probs[idx] * 0.5, 1.0), 0.3);
            } else {
                color.color.r = 0;
                color.color.g = 1;
                color.color.b = 0;
                color.color.a = 0.2;
            }
            ROS_INFO("obstacle (%d, %d): alpha - %f", i, j, color.color.a);
            color.id = collision_object.id;
            planning_scene.object_colors.push_back(color);

        }
    }

    planning_scene.is_diff = true;
    while (planning_scene_diff_publisher_.getNumSubscribers() < 1)
    {
        ros::WallDuration sleep_t(0.5);
        sleep_t.sleep();
    }
    // Publish the added object
    planning_scene_diff_publisher_.publish(planning_scene);
}

void FetchMoveBaseDemo::removeUncertainBoxes() {

    moveit_msgs::PlanningScene planning_scene;

    for (int i = 1; i < LabDomain::x_size - 1; ++i) {
        for (int j = 0; j < LabDomain::y_size; ++j) {

            moveit_msgs::CollisionObject collision_object;
            collision_object.header.frame_id = "/map";
            collision_object.id = "uncertain_box_" + std::to_string(i) + "_" + std::to_string(j);

            collision_object.operation = collision_object.REMOVE;
            planning_scene.world.collision_objects.push_back(collision_object);
        }
    }

    planning_scene.is_diff = true;
    while (planning_scene_diff_publisher_.getNumSubscribers() < 1)
    {
        ros::WallDuration sleep_t(0.5);
        sleep_t.sleep();
    }
    // Publish the added object
    planning_scene_diff_publisher_.publish(planning_scene);
}

void FetchMoveBaseDemo::pickUp() {
    const std::string TRAJECTORY = target_object_grasping_traj_name_;

    moveit::planning_interface::MoveGroup::Plan my_plan;
    moveit::planning_interface::MoveGroup move_group("arm_with_torso");
    moveit::planning_interface::MoveGroup gripper("gripper");

    move_group.setPlannerId("RRTConnectkConfigDefault");
    auto pcm = planning_scene_monitor::PlanningSceneMonitorPtr(
            new planning_scene_monitor::PlanningSceneMonitor("robot_description"));
    planning_scene_monitor::LockedPlanningSceneRO planning_scene(pcm);

    // This adds the safety box above the fetch base to avoid self-collisions.
    addBaseBox();

    vicon_object_server::VICONObjectClient client(true);
    Eigen::Affine3d pre_grasp = client.trajGetWaypoint(TRAJECTORY, 0);
    Eigen::Affine3d grasp = client.trajGetWaypoint(TRAJECTORY, 1);

    // Open the gripper before starting
    ROS_INFO("Opening gripper ...");
    control_msgs::GripperCommandGoal gripper_goal;
    gripper_goal.command.position = 0.115;
    gripper_goal.command.max_effort = 100.0;
    gripper_action_client_->sendGoal(gripper_goal);

    // Pregrasp
    ROS_INFO("Moving to pregrasp pose ...");
    geometry_msgs::PoseStamped target_pose_pre_grasp;
    tf::poseEigenToMsg(pre_grasp, target_pose_pre_grasp.pose);
    target_pose_pre_grasp.header.frame_id = "/vicon_object_server/" + target_object_name_;
    move_group.setPoseTarget(target_pose_pre_grasp);

    auto success = move_group.plan(my_plan);
    if (success == moveit::planning_interface::MoveItErrorCode::SUCCESS)
    {
        move_group.execute(my_plan);
    } else {
        return;
    }
    sleep(5.0);

    // Grasp
    ROS_INFO("Moving to grasping pose ...");
    move_group.clearPoseTargets();
    geometry_msgs::PoseStamped target_pose_grasp;
    tf::poseEigenToMsg(grasp, target_pose_grasp.pose);
    target_pose_grasp.header.frame_id = "/vicon_object_server/" + target_object_name_;
    move_group.setPoseTarget(target_pose_grasp);

    success = move_group.plan(my_plan);
    if (success == moveit::planning_interface::MoveItErrorCode::SUCCESS)
        move_group.execute(my_plan);
    else {
        return;
    }
    sleep(1.0);

    // Close the gripper
    ROS_INFO("Closing the gripper ...");
    gripper_goal.command.position = 0.0;
    gripper_action_client_->sendGoal(gripper_goal);
    sleep(2.0);

    move_group.attachObject(target_object_name_, "r_gripper_finger_link",
                            {"r_gripper_finger_link", "l_gripper_finger_link", "gripper_"
                                                                               "link"});
    client.objectDisable(target_object_name_);
    client.objectDisable(table_name_);
    sleep(1.0);


    // Pregrasp
    ROS_INFO("Moving to pregrasp pose ...");
    tf::poseEigenToMsg(pre_grasp, target_pose_pre_grasp.pose);
    target_pose_pre_grasp.header.frame_id = "/vicon_object_server/" + target_object_name_;
    move_group.setPoseTarget(target_pose_pre_grasp);

    success = move_group.plan(my_plan);
    if (success == moveit::planning_interface::MoveItErrorCode::SUCCESS)
        move_group.execute(my_plan);
    client.objectEnable(table_name_);

    move_group.detachObject(target_object_name_);
    client.objectEnable(target_object_name_);
}

void FetchMoveBaseDemo::executePolicy() {
    int num_obs = obstacle_names_.size();
    ROS_INFO("Number of obstacles: %d", num_obs);
    MyUtils::setDebugLevel(1);
    MyUtils::solver_mode = "success";
    POMDPDomainPtr domain = make_shared <LabDomain> (test_file_path_);

    ROS_INFO("Policy stay threshold: %f", policy_stay_th_);
    ROS_INFO("Horizon bound: %d", horizon_bound_);
    PolicySolverPtr solver = make_shared <PartialSolver> (policy_stay_th_);

    if (policy_ == nullptr) {
        policy_ = solver->solve(domain, 0, horizon_bound_, false);
    }

    init_probs = std::static_pointer_cast<LabDomain> (policy_->getDomain())->getObsProbs(
            policy_->getBelief(), 0
    );

    for (int i = 0; i < init_probs.size(); ++i) {
        ROS_INFO("Initial belief: %d - %f", i, init_probs[i]);
    }
    std::string input;
    ROS_INFO("Start?: yes / no");
    std::cin >> input;
    if (input == "no") {
        return;
    }

    std::vector<double> base_pose = getCurrentBasePose();
    double fetch_x = base_pose[0];
    double fetch_y = base_pose[1];
    double x_start = fetch_x + forword_dist_one_step_;
    double y_start = fetch_y - 0.05;

    for (int i = 1; i < LabDomain::x_size - 1; ++i) {
        for (int j = 0; j < LabDomain::y_size; ++j) {
            int idx = (i - 1) * LabDomain::y_size + j;
            ROS_INFO("Initial belief: (%d, %d) - %f", i, j, init_probs[idx]);
            double x = x_start + forword_dist_one_step_ * (i - 1) + forword_dist_one_step_ / 2;
            double y = y_start - forword_dist_one_step_ * j;
            ROS_INFO("Cell Center: (%f, %f)", x, y);
            uncertain_boxes.push_back(make_pair(x, y));
        }
    }

    replan_num_ += 1;
    if (policy_ != nullptr) {
        ROS_INFO("Start execution");
        bool success = execute(0, policy_, horizon_bound_);

        double stay_prob_computed = policy_->calculateStayProb();
        ROS_INFO (
                  "\nPolicy Stay Probability: %f",
                  stay_prob_computed
        );
        auto policy_info = PolicyNode::getPolicyInfo(policy_);
        ROS_INFO ("policy depth: %d", policy_info.first);

        ROS_INFO ("policy plan number: %d", policy_info.second);

        ROS_INFO ("Replanning number: %d", replan_num_);

        std::stringstream ss;
        PolicyNode::printPolicy(policy_, ss);
        if (success) {
            ROS_INFO("Success!");
        } else {
            ROS_INFO("Fail");
        }


        ofstream log_file;
        log_file.open ("/home/yuew/policy.out");
        log_file << ss.str() << endl;
        log_file.close();
    } else {
        ROS_WARN("Can not find valid policy");
    }
}

bool FetchMoveBaseDemo::execute(int step, PolicyNodePtr policy, int horizon_bound) {
    current_probs = std::static_pointer_cast<LabDomain> (policy->getDomain())->getObsProbs(
            policy->getBelief(), step
    );
    for (int i = 1; i < LabDomain::x_size - 1; ++i) {
        for (int j = 0; j < LabDomain::y_size; ++j) {
            int idx = (i - 1) * LabDomain::y_size + j;
            ROS_INFO("Belief at step %d: (%d, %d) - %f", step, i, j, current_probs[idx]);
        }
    }

    int action = policy->getAction();
    POMDPDomainPtr domain = policy->getDomain();
    if (policy->isGoal()) {
        ROS_INFO ("\nfinished execution");
        return true;
    }
    addUncertainBoxes();

    std::vector<double> base_pose = getCurrentBasePose();
    double x = base_pose[0];
    double y = base_pose[1];
    ROS_INFO("fetch: (%f, %f)", x, y);
    for (int i = 0; i < obstacle_names_.size(); ++i) {
        std::vector<double> obstacle_pose = getCurrentObjectPose(obstacle_names_[i]);
        ROS_INFO("obstacle %d: (%f, %f)", i, obstacle_pose[0], obstacle_pose[1]);
    }
    std::vector<double> target_object_pose = getCurrentObjectPose(target_object_name_);
    ROS_INFO("target object: (%f, %f)", target_object_pose[0], target_object_pose[1]);
    std::vector<double> table_pose = getCurrentObjectPose(table_name_);
    ROS_INFO("table: (%f, %f)", table_pose[0], table_pose[1]);
    double goal_x = table_pose[0] - table_offset_;
    ROS_INFO ("\nexecution in step %d: %s", step + 1, domain->getActionMeaning(action).c_str());
    addUncertainBoxes();

    std::string input;


    double new_x, new_y;
    switch (action) {
        case move_north_idx:
        {
            new_x = x;
            new_y = y - forword_dist_one_step_ - 0.13;
            break;
        }
        case move_south_idx:
        {
            new_x = x;
            new_y = y + forword_dist_one_step_;
            break;
        }
        case move_east_idx:
        {
            new_x = x + forword_dist_one_step_ + 0.1;
            new_y = y;
            break;
        }
        case move_west_idx:
        {
            new_x = x - forword_dist_one_step_ - 0.1;
            new_y = y;
            break;
        }
        case pick_up_idx:
        {
            std::vector<double> table_pose = getCurrentObjectPose(table_name_);
            new_x = table_pose[0] - table_offset_;
            new_y = table_pose[1];
            break;
        }
        default:
        {
            // for look actions
            new_x = x;
            new_y = y;
        }
    }
    int obs_received;
    if (new_x == x && new_y == y) {
        string look_direct;
        switch (action) {
            case look_north_idx:
                look_direct = "down";
                break;
            case look_south_idx:
                look_direct = "up";
                break;
            case look_east_idx:
                look_direct = "forward";
                break;
            case look_west_idx:
                look_direct = "backward";
                break;
            default:
                ROS_WARN("Unknown direction to observe");
                return false;
        }
        ROS_INFO("Continue?: yes / no");
        std::cin >> input;
        if (input == "no") {
            return false;
        }
        bool see_obs = observe(look_direct);
        if (see_obs) {
            obs_received = positive_observation;
            ROS_INFO("Observation received: obstacle observed");
        } else {
            obs_received = negative_observation;
            ROS_INFO("Observation received: no obstacle observed");
        }
    } else {
        ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", new_x, new_y);
        std::cin >> input;
        if (input == "yes") {
            goTo(new_x, new_y);
            addUncertainBoxes();
        } else {
            return false;
        }
        obs_received = trans_observation;
        if (action == pick_up_idx) {
            ROS_INFO("Pick up?: yes / no");
            std::cin >> input;
            if (input == "yes") {
                turn(0.0);
                addUncertainBoxes();
                removeUncertainBoxes();
                pickUp();
            }
            obs_received = pickup_positive_observation;
        }
    }
    string obs_received_str = domain->getObservationMeaning(obs_received);
    auto &child_nodes = policy->getChildNodes();
    if (child_nodes.find(obs_received) != child_nodes.end()) {
        PolicyNodePtr next = child_nodes[obs_received];
        ROS_INFO ("observation is covered by the partial policy: %s, proceed", (obs_received_str).c_str());
        return execute(step + 1, next, horizon_bound);
    }

    // this observation is not covered by the partial policy
    // should rerun partial policy generation
    ROS_INFO ("observation is not covered by the partial policy: %s, re-generate partial policy",
              domain->getObservationMeaning(obs_received).c_str());
    BeliefStatePtr new_b_next_vals = domain->getNextBelief(policy->getBelief(),
                                                           action, obs_received, step + 1);
    replan_num_ += 1;
    ROS_INFO("new next belief for obervation branch in step %d - %s:\n%s",
             step + 1,
             obs_received_str.c_str(),
             domain->beliefStateToString(new_b_next_vals, step + 1).c_str()
    );
    POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(step + 1, new_b_next_vals, action, obs_received);

    current_probs = std::static_pointer_cast<LabDomain> (domain)->getObsProbs(
            new_b_next_vals, step + 1
    );
    addUncertainBoxes();

    point start = MyUtils::now();
    PolicySolverPtr solver = make_shared <PartialSolver> (policy_stay_th_);
    PolicyNodePtr new_policy = solver->solve(new_domain, step + 1, horizon_bound, false);
    point end = MyUtils::now();
    ROS_INFO ("partial policy generation time : %fs", MyUtils::run_time(end - start));
    if (new_policy == nullptr) {
        return false;
    }
    policy->insertObservationBranch(obs_received, new_policy);
    return execute(step + 1, new_policy, horizon_bound);
}

void FetchMoveBaseDemo::executeDefaultPolicy() {
    
    std::string input;

    while (true) {
        std::vector<double> base_pose = getCurrentBasePose();
        double x = base_pose[0];
        double y = base_pose[1];
        ROS_INFO("fetch: (%f, %f)", x, y);
        for (int i = 0; i < obstacle_names_.size(); ++i) {
			std::vector<double> obstacle_pose = getCurrentObjectPose(obstacle_names_[i]);
			ROS_INFO("obstacle %d: (%f, %f)", i, obstacle_pose[0], obstacle_pose[1]);
		}
		std::vector<double> target_object_pose = getCurrentObjectPose(target_object_name_);
		ROS_INFO("target object: (%f, %f)", target_object_pose[0], target_object_pose[1]);
		std::vector<double> table_pose = getCurrentObjectPose(table_name_);
		ROS_INFO("table: (%f, %f)", table_pose[0], table_pose[1]);
        double goal_x = table_pose[0] - table_offset_;
        if (goal_x - x  < forword_dist_one_step_) {
            break;
        }

        // observe
        bool see_obs = observe("forward");
        if (see_obs) {
            see_obs = observe("down");
            if (see_obs) {
				ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", x - forword_dist_one_step_, y);
				std::cin >> input;
				if (input == "no") {
					break;
				}
                goTo(x - forword_dist_one_step_, y);
                ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", x - forword_dist_one_step_, y - forword_dist_one_step_);
				std::cin >> input;
				if (input == "no") {
					break;
				}
                goTo(x - forword_dist_one_step_, y - forword_dist_one_step_);
            } else {
				ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", x, y - forword_dist_one_step_);
				std::cin >> input;
				if (input == "no") {
					break;
				}
                goTo(x, y - forword_dist_one_step_);
            }
        } else {
            double forward_d = std::min(goal_x, forword_dist_one_step_);
            ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", x + forward_d, y);
				std::cin >> input;
				if (input == "no") {
					break;
				}
            goTo(x + forward_d, y);
        }

    }
    ROS_INFO("Move to table?: yes / no");
    std::cin >> input;
    if (input == "yes") {
		std::vector<double> base_pose = getCurrentBasePose();
        std::vector<double> table_pose = getCurrentObjectPose(table_name_);
        double goal_x = table_pose[0] - table_offset_;
        double goal_y = table_pose[1];
        ROS_INFO("Command: goTo(%f, %f),Continue?: yes / no", goal_x, goal_y);
		std::cin >> input;
		if (input == "yes") {
			goTo(goal_x, goal_y);
			turn(0.0);
		}
        
    }

    ROS_INFO("Pick up?: yes / no");
    std::cin >> input;
    if (input == "yes") {
        pickUp();
    }
}

void FetchMoveBaseDemo::goTo(double x, double y) {
    // get current base pose
    std::vector<double> base_pose = getCurrentBasePose();
    double x_pre = base_pose[0];
    double y_pre = base_pose[1];

    // first make a turn
    double dx = x - x_pre;
    double dy = y - y_pre;
    double yaw = tf2Atan2(dy,  dx);

    ROS_INFO("Robot base target yaw: %f", yaw);
    turn(yaw);

    printCurrentBasePose();

    // then forward
    forward(x, y);

    base_pose = getCurrentBasePose();
    double x_new = base_pose[0];
    double y_new = base_pose[1];
    if (std::abs(x - x_new) > xy_goal_tolerence_
        || std::abs(y - y_new) > xy_goal_tolerence_) {
        std::string input;
        ROS_INFO("Try again?: yes / no");
        std::cin >> input;
        if (input == "yes") {
            goTo(x, y);
        }
    }
}

void FetchMoveBaseDemo::fetchTurnFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr &feedback,
                                                  double yaw_goal) {
    if (current_probs.size() > 0) {
        addUncertainBoxes();
    }
    double yaw = tf2::getYaw(feedback->base_position.pose.orientation);
    if (std::abs(yaw - yaw_goal) < yaw_goal_tolerence_) {
        ROS_INFO("Current yaw (%f) reach target yaw (%f). Cancelling goal.", yaw, yaw_goal);
        move_base_client_.cancelGoal();
    }
}

void FetchMoveBaseDemo::turn(double yaw) {
    // get current base pose
    std::vector<double> base_pose = getCurrentBasePose();
    double x = base_pose[0];
    double y = base_pose[1];

    // transform yaw to quaternion
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();

    move_base_msgs::MoveBaseGoal goal;

    // prepare goal
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = x;
    goal.target_pose.pose.position.y = y;
    goal.target_pose.pose.orientation.x = q.getX();
    goal.target_pose.pose.orientation.y = q.getY();
    goal.target_pose.pose.orientation.z = q.getZ();
    goal.target_pose.pose.orientation.w = q.getW();

    ROS_INFO("Sending goal:\n\tx: %f, y: %f, yaw: %f, orientation: (%f, %f, %f, %f)]",
             goal.target_pose.pose.position.x, goal.target_pose.pose.position.y,
             tf2::getYaw(goal.target_pose.pose.orientation), goal.target_pose.pose.orientation.x,
             goal.target_pose.pose.orientation.y, goal.target_pose.pose.orientation.z,
             goal.target_pose.pose.orientation.w
    );
    move_base_client_.sendGoal(goal, MoveBaseClient::SimpleDoneCallback(),
                               MoveBaseClient::SimpleActiveCallback(),
                               boost::bind(&FetchMoveBaseDemo::fetchTurnFeedbackCallback, this,
                                           _1, yaw)
    );

    move_base_client_.waitForResult();

    if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("Robot base turned to target yaw");
    else
        ROS_INFO("Robot base stopped for some reason");
    // stop tracking goal
    move_base_client_.stopTrackingGoal();
}

void FetchMoveBaseDemo::forward(double d) {
    // get current base pose
    std::vector<double> base_pose = getCurrentBasePose();
    double x = base_pose[0];
    double y = base_pose[1];
    double yaw = base_pose[2];

    // calculate target location
    double dx = d * tf2Cos(yaw);
    double dy = d * tf2Sin(yaw);

    forward(x + dx, y + dy);
}

void FetchMoveBaseDemo::fetchForwadFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr &feedback,
                                                    double x_goal, double y_goal) {
    if (current_probs.size() > 0) {
        addUncertainBoxes();
    }
    double x = feedback->base_position.pose.position.x;
    double y = feedback->base_position.pose.position.y;
    if (std::abs(x - x_goal) < xy_goal_tolerence_
        && std::abs(y - y_goal) < xy_goal_tolerence_) {
        ROS_INFO("Current location (%f, %f) reach target location (%f, %f). Cancelling goal.",
            x, y, x_goal, y_goal);
        move_base_client_.cancelGoal();
    }
}

void FetchMoveBaseDemo::forward(double x, double y) {
    // get current base pose
    std::vector<double> base_pose = getCurrentBasePose();

    move_base_msgs::MoveBaseGoal goal;
    // prepare goal
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = x;
    goal.target_pose.pose.position.y = y;
    goal.target_pose.pose.orientation = base_to_map_tf_.transform.rotation;

    ROS_INFO("Sending goal:\n\tx: %f, y: %f, yaw: %f, orientation: (%f, %f, %f, %f)]",
             goal.target_pose.pose.position.x, goal.target_pose.pose.position.y,
             tf2::getYaw(goal.target_pose.pose.orientation), goal.target_pose.pose.orientation.x,
             goal.target_pose.pose.orientation.y, goal.target_pose.pose.orientation.z,
             goal.target_pose.pose.orientation.w
    );
    move_base_client_.sendGoal(goal, MoveBaseClient::SimpleDoneCallback(),
                               MoveBaseClient::SimpleActiveCallback(),
                               boost::bind(&FetchMoveBaseDemo::fetchForwadFeedbackCallback, this,
                                           _1, goal.target_pose.pose.position.x,
                                           goal.target_pose.pose.position.y
                               )
    );

    move_base_client_.waitForResult();

    if(move_base_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("Robot base move to target location");
    else
        ROS_INFO("Robot base stopped for some reason");
    // stop tracking goal
    move_base_client_.stopTrackingGoal();
}

void FetchMoveBaseDemo::printCurrentBasePose() {
    // get current base pose
    std::vector<double> base_pose = getCurrentBasePose();
    double x = base_pose[0];
    double y = base_pose[1];
    double yaw = base_pose[2];
    ROS_INFO(
            "Current Fetch Location:\n\tx: %f, y: %f, yaw: %f, orientation: (%f, %f, %f, %f)\n",
            x, y, yaw,
            base_to_map_tf_.transform.rotation.x, base_to_map_tf_.transform.rotation.y,
            base_to_map_tf_.transform.rotation.z, base_to_map_tf_.transform.rotation.w
    );
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fetch_move_base_demo");
    ros::AsyncSpinner spinner(4);
    spinner.start();

    FetchMoveBaseDemo demo;
    while (ros::ok()) {
        demo.printCurrentBasePose();

        ROS_INFO("Please input command (run_default, execute, move x y, turn yaw, forward d or exit)");
        std::string input;
        std::cin >> input;
        if (input == "exit") {
            ROS_INFO("Stop");
            break;
        } else if (input == "run_default") {
            demo.executeDefaultPolicy();
        } else if (input == "execute") {
            demo.executePolicy();
        } else if (input == "move") {
            std::cin >> input;
            double x = std::stod(input);
            std::cin >> input;
            double y = std::stod(input);
            demo.goTo(x, y);
        } else if (input == "turn") {
            std::cin >> input;
            double yaw = std::stod(input);
            demo.turn(yaw);
        } else if (input == "forward") {
            std::cin >> input;
            double d = std::stod(input);
            demo.forward(d);
        } else {
            ROS_INFO("Unknown command: exit");
            break;
        }
    }
    spinner.stop();
    ros::shutdown();
    return 0;
}
