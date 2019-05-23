//
// Created by Yue Wang on 6/16/18.
//

#ifndef POLICY_SYNTHESIZER_LAB_DOMAIN_H
#define POLICY_SYNTHESIZER_LAB_DOMAIN_H

#include "corridor_domain.h"
#include <unordered_map>

using namespace std;

class LabDomain : public CorridorDomain {
public:
    LabDomain (string test_file_path);
    LabDomain (int s, BeliefStatePtr init_b,
               int a_s, int o_s, unordered_map<int, int> init_b_to_g_map);
    POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr new_init_b, int a_s, int o_s);
    z3::expr genTransCond(z3::context &ctx, int step);
    z3::expr genGoalCond(z3::context & ctx, int step);

    string getActionMeaning(int action_id);

    vector<double> getObsProbs(BeliefStatePtr b, int step);
    void generateRandomInstances(string test_file_dir, int num);

protected:

    void init();

    int getNextLoc(int action_idx, int rob_loc_pre);

    void updateGroupSplitIds(unordered_map<int, string> &group_split_ids, int step);
};

#endif //POLICY_SYNTHESIZER_LAB_DOMAIN_H
