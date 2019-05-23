//
// Created by Redwan Newaz on 2019-04-27.
//

#ifndef T_ASE_KITCHEN_BELIEF_H
#define T_ASE_KITCHEN_BELIEF_H
#include "pomdps/pomdp_domain.h"
#include <algorithm>


class kitchen_belief : public POMDPDomain {

public:

    kitchen_belief(): POMDPDomain(0, 0, 0){}
    kitchen_belief (int s, BeliefStatePtr init_b,
                    int a_s, int o_s, unordered_map<int, int> init_b_to_g_map, bool run_init = true);
    void init(bool run_init);
    string beliefStateToString(BeliefStatePtr b, int step);
    string getActionMeaning(int action_id);
    string getObservationMeaning(int observation_id);
    vector<pair<int, double>> getObservationDistribution(int action_id,
                                                         BeliefStatePtr b_pre, int step);
    int observe(BeliefStatePtr b, int action_id, int step);
    void generateRandomInstances(string test_file_dir, int num);

    void computeAlphaVector(PolicyNodePtr policy);
    double computeBeliefValue(BeliefStatePtr b,
                              unordered_map<int, double> &alpha, int step);

    unordered_map<int, bool> getAvailableActions();

    BeliefStatePtr getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
                                       int obs_id, int step);
    bool isGoal(BeliefStatePtr b, int step);
protected:
    pair<int, int> getXY(int loc_idx);
    int getLocIdx(int x, int y);
    int getStateIdx(int rob_loc_idx, int obs_locs_idx);
    int getRobotLocIdx(int state_idx);
    int getObsLocsIdx(int state_idx);

    virtual int getNextLoc(int action_idx, int rob_loc_pre);
    bool in_collision(int rob_loc, const vector<int> &obs_locs);
    void genObstacleLocsList(int obs_idx, vector<int> &obs_locs_temp, int loc_idx,
                             const vector<int> &obs_all_locs,
                             vector<vector<int>> &obs_locs_list);

    pair<vector<double>, vector<double>> extractProbs(BeliefStatePtr b, int step);
    void updateBeliefGroupMap(int step);
    virtual void updateGroupSplitIds(unordered_map<int, string> &group_split_ids, int step);
    void updateGroupSplitIdsForAction(int action_idx, unordered_map<int, string> &group_split_ids, int step);
    void printBeliefGroupInfo(int step);
    BeliefStatePtr filterZeroProb(BeliefStatePtr b, int step);

    static double p_fp;
    static double p_fn;
    static double p_succ;
    static int x_size;
    static int y_size;
    static double delta;

    static int rob_locs_size;
    static int ready_robot_loc;
    static int unsafe_robot_loc;
    static int goal_robot_loc;
    static int num_obs;

    static vector<int> obs_locs;

    vector<int> avoid_regions;
    vector<int> obs_all_locs;
    vector<vector<int>> obs_locs_list;

    int see_obs_num;

    BeliefStatePtr init_belief;

    vector<vector<int>> belief_group_size;
    vector<unordered_map<int, int>> belief_to_group_maps;



    const double reach_th = 0.8;

    int init_robot_loc;

};


#endif //T_ASE_KITCHEN_BELIEF_H
