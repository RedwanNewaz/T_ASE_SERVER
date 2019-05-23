//
// Created by Redwan Newaz on 2019-04-27.
//

#ifndef T_ASE_KITCHEN_MAIN_H
#define T_ASE_KITCHEN_MAIN_H

#include "kitchen/kitchen_belief.h"
#include <cassert>
#include <iterator>

class kitchen_main: public kitchen_belief {
    friend class kitchen_tau;
public:
    kitchen_main(string test_file_path, bool run_init = true);
    kitchen_main(int s, BeliefStatePtr init_b,
                 int a_s, int o_s,
                 unordered_map<int, int> init_b_to_g_map, bool run_init = true);
    POMDPDomainPtr createNewPOMDPDomain(int s, BeliefStatePtr new_init_b, int a_s, int o_s);

    z3::expr genTransCond(z3::context &ctx, int step);
    z3::expr genInitCond(z3::context &ctx);
    z3::expr genGoalCond(z3::context & ctx, int step);
    vector<z3::expr> getBeliefStateVarList(z3::context &ctx, int step);
    BeliefStatePtr getBeliefState(z3::context &ctx, ModelPtr model, int step);
    void publish(BeliefStatePtr,int);

    virtual ~kitchen_main();

private:

    z3::expr genBeliefSum(z3::context &ctx, int step);
    z3::expr genMoveActionTransCond(z3::context &ctx, int action_idx, int step);
    z3::expr genMoveActionPreCond(z3::context &ctx, int action_idx, int step);
    z3::expr genLookActionTransCond(z3::context &ctx, int action_idx, int step);
    z3::expr genLookActionPreCond(z3::context &ctx, int action_idx, int step);
    z3::expr genPickUpTransCond(z3::context &ctx, int action_idx, int step);
    z3::expr genMoveSafeCond(z3::context &ctx, int step);
    z3::expr genPickUpSafeCond(z3::context &ctx, int step);

protected:
public:
    BeliefStatePtr getNextBeliefDirect(BeliefStatePtr b_pre, int action_id, int obs_id, int step) override;
    vector<pair<int, double>> getObservationDistribution(int action_id, BeliefStatePtr b_pre, int step) override;




};


#endif //T_ASE_KITCHEN_MAIN_H
