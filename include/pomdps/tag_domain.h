//
// Created by Yue Wang on 1/20/18.
//

#ifndef POLICY_SYNTHESIZER_TAG_DOMAIN_H
#define POLICY_SYNTHESIZER_TAG_DOMAIN_H

#include "pomdp_domain.h"

class TagDomain : public  POMDPDomain {
public:
    TagDomain (string test_file_path);
    TagDomain (int s, BeliefStatePtr init_b, int a_s, int o_s);
    POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr init_b,
                                        int a_s, int o_s);

    z3::expr genInitCond(z3::context &ctx);
    z3::expr genTransCond(z3::context &ctx, int step);
    z3::expr genGoalCond(z3::context &ctx, int step);

    vector<z3::expr> getBeliefStateVarList(z3::context &ctx, int step);
    string beliefStateToString(BeliefStatePtr b, int step);
    BeliefStatePtr getBeliefState(z3::context &ctx, ModelPtr model, int step);
    string getActionMeaning(int action_id);
    string getObservationMeaning(int observation_id);
    vector<pair<int, double>> getObservationDistribution(
            int action_id, BeliefStatePtr b_pre, int step);

    int observe(BeliefStatePtr b, int action_id, int step);
    void generateRandomInstances(string test_file_dir, int num);

    void computeAlphaVector(PolicyNodePtr policy);

    double computeBeliefValue(BeliefStatePtr b,
                              unordered_map<int, double> &alpha, int step);

    unordered_map<int, bool> getAvailableActions();

    BeliefStatePtr getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
                                 int obs_id, int step);

    bool isGoal(BeliefStatePtr b, int step);

private:
    const int belief_size = 29;
    const int no_obs = 0;
    const int yes_obs = 1;
    const int tag_success_obs = 2;

    const int north_idx = 1;
    const int south_idx = 2;
    const int east_idx = 3;
    const int west_idx = 4;
    const int stay_idx = 5;
    const int tag_idx = 6;

    const double yes_th = 0.0001;
    const double zero_th = 0.001;
    const double gamma = 0.95;

    vector<unordered_map<int, bool>> rob_locs_set;

    static double tag_th;
    int rob_init_loc;
    static int curr_target_loc;

    BeliefStatePtr init_belief;


    z3::expr genMoveTransCond(z3::context &ctx, int action_id, int step);
    z3::expr genTagTransCond(z3::context &ctx, int step);
    z3::expr genBeliefNoChange(z3::context &ctx, int step);
    int distance(int l1, int l2);
    int getX(int idx);
    int getY(int idx);

    int getStateIdx(int r, int i);
    vector<double> getNextTargetLocDist(int r_loc, int t_loc);
    int getNextLoc(int action_id, int curr_loc);

    z3::expr genBeliefSum(z3::context &ctx, int step);

};

#endif //POLICY_SYNTHESIZER_TAG_DOMAIN_H
