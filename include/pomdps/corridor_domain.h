#ifndef CORRIDOR_DOMAIN_H
#define CORRIDOR_DOMAIN_H

#include "pomdp_domain.h"
#include <unordered_map>
#include <chrono>
#include <thread>

using namespace std;
class CorridorDomain : public POMDPDomain {
public:
	CorridorDomain (string test_file_path, bool run_init = true);
    CorridorDomain (int s, BeliefStatePtr init_b,
                    int a_s, int o_s, unordered_map<int, int> init_b_to_g_map, bool run_init = true);
    POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr new_init_b, int a_s, int o_s);
	z3::expr genTransCond(z3::context &ctx, int step);
    z3::expr genInitCond(z3::context &ctx);
    z3::expr genGoalCond(z3::context & ctx, int step);

    vector<z3::expr> getBeliefStateVarList(z3::context &ctx, int step);
    BeliefStatePtr getBeliefState(z3::context &ctx, ModelPtr model, int step);
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


    static double p_fp;
    static double p_fn;
    static double p_succ;
	static int x_size;
	static int y_size;
	static double delta;

protected:
    void init(bool run_init);

    BeliefStatePtr filterZeroProb(BeliefStatePtr b, int step);

    z3::expr genBeliefSum(z3::context &ctx, int step);

	z3::expr genMoveActionTransCond(z3::context &ctx, int action_idx, int step);
    z3::expr genMoveActionPreCond(z3::context &ctx, int action_idx, int step);
    z3::expr genLookActionTransCond(z3::context &ctx, int action_idx, int step);
    z3::expr genLookActionPreCond(z3::context &ctx, int action_idx, int step);
    z3::expr genPickUpTransCond(z3::context &ctx, int action_idx, int step);

    z3::expr genMoveSafeCond(z3::context &ctx, int step);
    z3::expr genPickUpSafeCond(z3::context &ctx, int step);


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


#endif
