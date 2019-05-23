//
// Created by Yue Wang on 1/12/18.
//

#ifndef POLICY_SYNTHESIZER_PLAN_H
#define POLICY_SYNTHESIZER_PLAN_H

#include "my_utils.h"
#include "partial_policy.h"

#include <memory>
#include <vector>
#include <z3++.h>


using namespace std;

class POMDPDomain;
typedef shared_ptr <POMDPDomain> POMDPDomainPtr;

class Plan;
typedef shared_ptr <Plan> PlanPtr;

class Plan {
public:
    Plan(z3::context &ctx, ModelPtr model, POMDPDomainPtr domain, int s, int h);
    z3::expr blockPlanPrefixConstraints(z3::context &ctx, int prefix_bound);
    int getAction(unsigned step);
    int getObservation(unsigned step);
    BeliefStatePtr getBelief(unsigned step);
    const vector<pair<int, double>> &getObservationDist(unsigned step);
    void printPlan(POMDPDomainPtr domain);
    int getPlanHorizon() {
        return horizon_bound;
    }
private:
    int start_step;
    int horizon_bound;
    vector<int> action_list;
    vector<int> observation_list;
    vector<BeliefStatePtr> belief_list;
    vector<vector<pair<int, double>>> observation_dist_list;
};
#endif //POLICY_SYNTHESIZER_PLAN_H
