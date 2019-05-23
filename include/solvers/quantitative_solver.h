//
// Created by Yue Wang on 9/24/18.
//

#ifndef BOOLEAN_POMDP_SYNTHESIZER_QUANTITATIVE_SOLVER_H
#define BOOLEAN_POMDP_SYNTHESIZER_QUANTITATIVE_SOLVER_H

#include "partial_solver.h"

class QuantitativeSolver : public  PartialSolver {
public:
    QuantitativeSolver (): PartialSolver(1.0) {
    }

    PolicyNodePtr solve(POMDPDomainPtr domain, int start_step, int max_k, bool run);

    PolicyNodePtr quantitativePolicySynthesis(POMDPDomainPtr domain,
                                              double stay_th, int start_step, int horizon_bound);

    double policyImprovement(const vector<PolicyNodePtr> &alpha_vectors_pre,
                             vector<PolicyNodePtr> &alpha_vectors_new,
                             PolicyNodePtr policy, int horizon_bound);
};

#endif //BOOLEAN_POMDP_SYNTHESIZER_QUANTITATIVE_SOLVER_H
