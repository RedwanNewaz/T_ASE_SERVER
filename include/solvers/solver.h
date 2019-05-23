#ifndef SOLVER_H
#define SOLVER_H

#include <memory>

#include "partial_policy.h"
#include "pomdps/pomdp_domain.h"

class Solver;
typedef shared_ptr <Solver> PolicySolverPtr;


class Solver
		: public enable_shared_from_this<Solver> {
public:
//	Solver(){};
	virtual PolicyNodePtr solve (POMDPDomainPtr domain, int start_step, int max_k, bool run = true) = 0;
	int one_run_planning_num;
	double solver_checking_time;
	int num_plan_checked;
    double synthesis_time;
    string exec_info;
    bool success;
};

#endif
