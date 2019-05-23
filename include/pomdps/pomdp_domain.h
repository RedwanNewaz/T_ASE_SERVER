#ifndef POMDP_DOMAIN_H
#define POMDP_DOMAIN_H

#include <z3++.h>
#include <vector>
#include <memory>
#include <unordered_map>

#include "belief_state.h"
#include "my_utils.h"
#include "partial_policy.h"


using namespace std;

class POMDPDomain;
typedef shared_ptr <POMDPDomain> POMDPDomainPtr;

class PolicyNode;
typedef shared_ptr <PolicyNode> PolicyNodePtr;

class Plan;
typedef shared_ptr <Plan> PlanPtr;

class Solver;
typedef shared_ptr <Solver> PolicySolverPtr;

enum {
    move_north_idx = 1,
    move_south_idx = 2,
    move_west_idx = 3,
    move_east_idx = 4,
    look_north_idx = 5,
    look_south_idx = 6,
    look_west_idx = 7,
    look_east_idx = 8,
    pick_up_left_hand_idx = 9,
    pick_up_right_hand_idx = 10,
};
const int pick_up_idx = pick_up_right_hand_idx;

enum {
    trans_observation = 1,
    positive_observation = 2,
    negative_observation = 3,
    pickup_positive_observation = 4,
    pickup_negative_observation = 5,
};


class POMDPDomain:
        public enable_shared_from_this<POMDPDomain> {
public:
	POMDPDomain(){start_step=a_start=o_start=0;};
    POMDPDomain (int s, int a_s, int o_s);
    BeliefStatePtr getNextBelief(BeliefStatePtr b_pre, int action_id,
                           int obs_id, int step);


    virtual POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr init_b,
                                                int a_s, int o_s) = 0;
    virtual z3::expr genTransCond(z3::context &ctx, int step) = 0;
	virtual z3::expr genGoalCond(z3::context &ctx, int step) = 0;
    virtual z3::expr genInitCond(z3::context &ctx) = 0;
    virtual vector<z3::expr> getBeliefStateVarList(z3::context &ctx, int step) = 0;
    virtual BeliefStatePtr getBeliefState(z3::context &ctx, ModelPtr model, int step) = 0;
    virtual string beliefStateToString(BeliefStatePtr b, int step) = 0;
	virtual string getActionMeaning(int action_id) = 0;
	virtual string getObservationMeaning(int observation_id) = 0;
    virtual vector<pair<int, double>> getObservationDistribution(
            int action_id, BeliefStatePtr b_pre, int step) = 0;
    virtual int observe(BeliefStatePtr b, int action_id, int step) = 0;
    virtual void generateRandomInstances(string test_file_dir, int num) = 0;

    PolicyNodePtr checkPolicy(BeliefStatePtr b, PolicyNodePtr policy, int start_step, int horizon_bound);

    PolicyNodePtr backup(const vector<PolicyNodePtr> &alpha_vectors_pre,
                         vector<PolicyNodePtr> &alpha_vectors_new,
                         PolicyNodePtr policy, PolicySolverPtr solver, int step, int horizon_bound);

    virtual void computeAlphaVector(PolicyNodePtr policy) = 0;
    virtual double computeBeliefValue(BeliefStatePtr b,
                                      unordered_map<int, double> &alpha, int step) = 0;

    virtual unordered_map<int, bool> getAvailableActions() = 0;
    virtual BeliefStatePtr getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
                                               int obs_id, int step) = 0;
    virtual bool isGoal(BeliefStatePtr b, int step)  = 0;

    static double epsilon;
    virtual void publish(BeliefStatePtr, int){};

protected:
    int start_step;
    int a_start;
    int o_start;

private:
    const bool check = false;
};

#endif
