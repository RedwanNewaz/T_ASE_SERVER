//
// Created by Yue Wang on 1/12/18.
//

#ifndef POLICY_SYNTHESIZER_PARTIAL_SOLVER_H
#define POLICY_SYNTHESIZER_PARTIAL_SOLVER_H

#include "plan.h"
#include "solver.h"
#include <unordered_map>
#define debug(x) std::cout<<blue <<x<<"\n"

#define SAFETY_CHECK true
#define MEMOIZATION true

class PartialSolver : public Solver {
public:
    PartialSolver (double stay_th, bool dp =false):enable_dp(dp) {
        stay_threshold = stay_th;
        one_run_planning_num = 0;
        solver_checking_time = 0.0;
        num_plan_checked = 0;
        synthesis_time = 0.0;
        exec_info = "";
        success = false;
    }
    PolicyNodePtr solve(POMDPDomainPtr domain, int start_step, int max_k, bool run);

    PolicyNodePtr partialPolicySynthesis(POMDPDomainPtr domain,
                                         double stay_th, int start_step, int horizon_bound);

    pair<PolicyNodePtr, z3::expr> partialPolicyGeneration(
            z3::context &ctx, POMDPDomainPtr domain, PlanPtr plan,
            double stay_th, int step, int horizon_bound);
    bool execute(int step, PolicyNodePtr policy, int horizon_bound);
    double stay_threshold;

protected:
    stringstream ss;

    char msgBuffer[MyUtils::DEBUG_BUFFER_SIZE];
    unordered_map<BeliefStatePtr, PolicyNodePtr>belief_to_pi;
private:
    bool enable_dp;



};

#endif //POLICY_SYNTHESIZER_PARTIAL_SOLVER_H
