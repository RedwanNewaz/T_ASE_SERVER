//
// Created by Redwan Newaz on 2018-12-23.
//

#ifndef BOOLEAN_POMDP_SYNTHESIZER_NAVIGATION_DOMAIN_H
#define BOOLEAN_POMDP_SYNTHESIZER_NAVIGATION_DOMAIN_H

#include "pomdp_domain.h"
#include "z3++.h"
#include "../../../../../CLionProjects/PolicySynthesis/Navigation/include/GridWorld.h"
#include <unordered_map>
#include <algorithm>
#include <functional>

#define debug(x) std::cout<<x<<std::endl
using namespace std;

class navigation_domain :public POMDPDomain {

public:
    navigation_domain(int s, int a_s, int o_s);

    explicit navigation_domain(std::shared_ptr<GridWorld>_grid);


    POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr init_b, int a_s, int o_s) override;

    z3::expr genTransCond(z3::context &ctx, int step) override;

    z3::expr genGoalCond(z3::context &ctx, int step) override;

    z3::expr genInitCond(z3::context &ctx) override;

    vector<z3::expr> getBeliefStateVarList(z3::context &ctx, int step) override;

    BeliefStatePtr getBeliefState(z3::context &ctx, ModelPtr model, int step) override;

    string beliefStateToString(BeliefStatePtr b, int step) override;

    string getActionMeaning(int action_id) override;

    string getObservationMeaning(int observation_id) override;

    vector<pair<int, double>> getObservationDistribution(int action_id, BeliefStatePtr b_pre, int step) override;

    int observe(BeliefStatePtr b, int action_id, int step) override;

    void generateRandomInstances(string test_file_dir, int num) override;

    void computeAlphaVector(PolicyNodePtr policy) override;

    double computeBeliefValue(BeliefStatePtr b, unordered_map<int, double> &alpha, int step) override;

    unordered_map<int, bool> getAvailableActions() override;

    BeliefStatePtr getNextBeliefDirect(BeliefStatePtr b_pre, int action_id, int obs_id, int step) override;

    bool isGoal(BeliefStatePtr b, int step) override;

private:
    std::shared_ptr<GridWorld>m_grid;
    BeliefStatePtr init_belief;

    vector<vector<int>> belief_group_size;
    vector<unordered_map<int, int>> belief_to_group_maps;

//STEP: define hyper parameters
    const double reach_th   = 0.8;
    const double p_fp       = 0.02;
    const double p_fn       = 0.05;
    const double p_succ     = 0.98;
//TODO: get the parameters from test file
    const double delta      = 0.00;
    const int num_obs       = 0;
    const double epsilon    = 0.0;
    int init_robot_loc;

protected:
    int nCr(int n, int k);
    vector<vector<int>> listCombination(int n, int r);

};


#endif //BOOLEAN_POMDP_SYNTHESIZER_NAVIGATION_DOMAIN_H
