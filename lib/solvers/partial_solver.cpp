//
// Created by Yue Wang on 1/12/18.
// Modified by Redwan Newaz on 04/29/2019
//


#ifndef PARTIAL_SOLVER_CC
#define PARTIAL_SOLVER_CC

#include "solvers/partial_solver.h"
#include "my_utils.h"
#include <string>
#include <fstream>
//#define WRITE_POLICY true
void elapsed_time(){
    static auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d = t2-t1;
    Color::Modifier red(Color::FG_RED);
    Color::Modifier blue(Color::FG_BLUE);
    Color::Modifier def(Color::FG_DEFAULT);
    std::cout << blue
              << "ELAPSED TIME "
              <<def
              <<"..("<<red
              << std::chrono::duration_cast<std::chrono::seconds>(d).count()
              <<def
              << " s)..\r" << std::flush;

}

// b1:x,y:p;o1:x,y:p;o2:x,y:p;...;a1:look east;z1:p


PolicyNodePtr PartialSolver::solve(POMDPDomainPtr domain, int start_step, int max_k, bool run) {

    point start = MyUtils::now();
    PolicyNodePtr policy = partialPolicySynthesis(domain, stay_threshold, start_step, max_k);
    point end = MyUtils::now();
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "partial policy generation time : %fs", MyUtils::run_time(end - start));
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    one_run_planning_num = 1;
    if (policy != nullptr && run) {

#ifdef WRITE_POLICY
        cout<<"policy tree depth "<<PolicyNode::getPolicyInfo(policy).first <<"\t";
        cout <<"path num "<<PolicyNode::getPolicyInfo(policy).second<<endl;
        ofstream myfile ("result/policy.txt");
        policy->writePolicyFile(policy,myfile);
        myfile.close();
#endif

        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                 "start execution:");
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        success = execute(0, policy, max_k);
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ZERO_MSG) {
            PolicyNode::printPolicy(policy, ss);
        }
        double stay_prob_computed = policy->calculateStayProb();
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "\nPolicy Stay Probability: %f",
                  stay_prob_computed
        );
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        auto policy_info = PolicyNode::getPolicyInfo(policy);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "policy depth: %d", policy_info.first);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "policy plan number: %d", policy_info.second);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        if (!success) {
            policy = nullptr;
        }
    }
    exec_info = ss.str();
    return policy;

}

bool PartialSolver::execute(int step, PolicyNodePtr policy, int horizon_bound) {
    POMDPDomainPtr domain = policy->getDomain();


    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "\ncurrent belief: %s",  domain->beliefStateToString(policy->getBelief(), step).c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer, MyUtils::LEVEL_ONE_MSG);

    int action = policy->getAction();
    if (policy->isGoal()) {


        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "\nfinished execution");
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        return true;
    }

    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "\nexecution in step %d: %s", step + 1, domain->getActionMeaning(action).c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    // observation dist
    const vector<pair<int, double>> &observation_dist = domain->getObservationDistribution(action, policy->getBelief(), step + 1);
    snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
             "observation distribution:");
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    for (auto o_p: observation_dist) {
        ss << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") ";
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") ";
        }
    }
    ss << endl;
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << endl;
    }
    // receive observation
//     int obs_received = MyUtils::sampleObservation(policy->getObservationDist());
    int obs_received = domain->observe(policy->getBelief(), action, step);
    string obs_received_str = domain->getObservationMeaning(obs_received);
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "observation received: %s", obs_received_str.c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    auto &child_nodes = policy->getChildNodes();
    if (child_nodes.find(obs_received) != child_nodes.end()) {
        PolicyNodePtr next = child_nodes[obs_received];
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "observation is covered by the partial policy: %s, proceed", (obs_received_str).c_str());
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        return execute(step + 1, next, horizon_bound);
    }
    // this observation is not covered by the partial policy
    // should rerun partial policy generation
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "observation is not covered by the partial policy: %s, re-generate partial policy",
              domain->getObservationMeaning(obs_received).c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    BeliefStatePtr new_b_next_vals = domain->getNextBelief(policy->getBelief(),
                                                            action, obs_received, step + 1);
    num_plan_checked += 1;
    snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
             "new next belief for obervation branch in step %d - %s:\n%s",
             step + 1,
             obs_received_str.c_str(),
             domain->beliefStateToString(new_b_next_vals, step + 1).c_str()
    );
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(step + 1, new_b_next_vals, action, obs_received);

    point start = MyUtils::now();
    PolicyNodePtr new_policy = partialPolicySynthesis(new_domain,
                                                      stay_threshold, step + 1, horizon_bound + step + 1);
    ++ one_run_planning_num;
    point end = MyUtils::now();
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "partial policy generation time : %fs", MyUtils::run_time(end - start));
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    if (new_policy == nullptr) {
        return false;
    }
    policy->insertObservationBranch(obs_received, new_policy);
    return execute(step + 1, new_policy, horizon_bound);
}

PolicyNodePtr PartialSolver::partialPolicySynthesis(POMDPDomainPtr domain,
                                                    double stay_th, int start_step,
                                                    int horizon_bound) {

    if (horizon_bound < start_step) {
        return nullptr;
    }
    z3::context ctx;

    z3::solver solver(ctx);

    //TODO - track unsafe belief and return null policy if there exist unsafe belief

    z3::expr init_cond = domain->genInitCond(ctx);
    MyUtils::printZ3Expr("initial condition:", init_cond, MyUtils::LEVEL_TWO_MSG);
    solver.add(init_cond);

    z3::expr_vector learned_constraints(ctx);
    for (int k = start_step; k <= horizon_bound; ++k) {
        elapsed_time();
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "current horizon: %d", k);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        // STEP add transition condition at step k
        if (k > start_step) {
            const z3::expr &trans_k = domain->genTransCond(ctx, k);
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "transition condition for step %d:", k);
            MyUtils::printZ3Expr(msgBuffer, trans_k, MyUtils::LEVEL_TWO_MSG);
            solver.add(trans_k);
        }
        const z3::expr &goal_k = domain->genGoalCond(ctx, k);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "goal condition for step %d:", k);
        MyUtils::printZ3Expr(msgBuffer, goal_k, MyUtils::LEVEL_TWO_MSG);
        // STEP incremental constraints - goal and learned constraints
        solver.push();
        solver.add(goal_k);
        solver.add(mk_and(learned_constraints));
        while (true) {
            z3::check_result  ret;
            ModelPtr model;
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "Begin Z3 solving ...");
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
            point start = MyUtils::now();
            if (MyUtils::solver_mode == "noinc") {
                // this is for non-inc
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "use non-incremental solving");
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                z3::solver new_solver(solver.ctx());
                new_solver.add(mk_and(solver.assertions()));
                ret = new_solver.check();
                if (ret == z3::sat) {
                    model = make_shared<z3::model> (new_solver.get_model());
                }
            } else {
                ret = solver.check();
                if (ret == z3::sat) {
                    model = make_shared<z3::model> (solver.get_model());
                }
            }
            point end = MyUtils::now();
            solver_checking_time += MyUtils::run_time(end - start);
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "z3 checking time: %fs", MyUtils::run_time(end - start));
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
            num_plan_checked += 1;

            if (ret == z3::unsat) {
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "No candidate plan found in step: %d, z3 returns UNSAT", k);
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                break;
            } else if (ret == z3::unknown) {
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "No candidate plan found in step: %d, z3 returns UNKNOWN", k);
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                break;
            } else if (ret == z3::sat) {
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "Found a candidate plan in step: %d", k);
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                PlanPtr plan = make_shared <Plan> (ctx, model, domain,
                                                   start_step, k);
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "The candidate plan is: ");
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                plan->printPlan(domain);


                auto gen_ret = partialPolicyGeneration(ctx, domain, plan, stay_th, start_step, horizon_bound);
                PolicyNodePtr policy = gen_ret.first;
                if (policy != nullptr) {
                    // find a valid policy
                    MyUtils::printDebugMsg("FOUND VALID POLICY ", MyUtils::LEVEL_TWO_MSG);
                    return policy;
                }
                // add additional constraints
                solver.add(gen_ret.second);
                learned_constraints.push_back(gen_ret.second);
//                debug(learned_constraints);
            } else {
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "unknown z3 return result");
                MyUtils::printErrorMsg(msgBuffer);
                break;
            }

        }
        // pop additional constraints
        solver.pop();
    }
    return nullptr;
}

pair<PolicyNodePtr, z3::expr> PartialSolver::partialPolicyGeneration(
        z3::context &ctx, POMDPDomainPtr domain, PlanPtr plan,
        double stay_th, int step, int horizon_bound) {
    BeliefStatePtr b_pre = plan->getBelief(step);
    int next_step = step + 1;

    if (step == plan->getPlanHorizon()) {
        // reach terminal state
        PolicyNodePtr policy =  make_shared<PolicyNode>(
                b_pre,
                0,
                domain,
                next_step,
                true);
        return make_pair(policy, ctx.bool_val(true));
    }



    int a_next = plan->getAction(next_step);
    int o_next = plan->getObservation(next_step);
    const auto &observation_dist_origin = plan->getObservationDist(next_step);
    auto observation_dist = plan->getObservationDist(next_step);

    PolicyNodePtr policy = make_shared<PolicyNode>(b_pre, a_next,
                                                   domain, next_step);

    auto gen_ret = partialPolicyGeneration(ctx, domain, plan, stay_th, next_step, horizon_bound);
    PolicyNodePtr p_next = gen_ret.first;
    if (p_next == nullptr) {
        return gen_ret;
    }
    policy->insertObservationBranch(o_next, p_next);


    double p_stay_total = 0.0;
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "try to complete the partial policy for action %s in step: %d",
              domain->getActionMeaning(a_next).c_str(), next_step
    );
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    double stay_th_temp = stay_th;
    //TODO importance sampling using MEMOIZATION
    while (true) {
        double o_next_prob = 0.0;
        // get observation probability
        for (int j = 0; j < observation_dist.size(); ++j) {
            int o = observation_dist[j].first;
            if (o == o_next) {
                o_next_prob = observation_dist[j].second;
            }
        }
        if (o_next_prob < MyUtils::DOUBLE_TOLENRANCE) {
            snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                     "get an observation with 0 proabibility");
            MyUtils::printErrorMsg(msgBuffer);
        }
        // update observation probability dist
        int remaining_obs_num = 0;
        vector<pair<int, double>> observation_dist_temp;
        for (int j = 0; j < observation_dist.size(); ++j) {
            int o = observation_dist[j].first;
            if (o != o_next) {
                double o_prob = observation_dist[j].second;
                observation_dist_temp.push_back(make_pair(o, o_prob / (1 - o_next_prob)));
            }
        }
        observation_dist = observation_dist_temp;
        // STEP update p_stay
        double o_next_origin = 0.0;
        for (int j = 0; j < observation_dist_origin.size(); ++j) {
            int o = observation_dist_origin[j].first;
            if (o == o_next) {
                o_next_origin = observation_dist_origin[j].second;
            }
        }
        if (o_next_origin < MyUtils::DOUBLE_TOLENRANCE) {
            snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                     "get an observation with 0 proabibility");
            MyUtils::printErrorMsg(msgBuffer);
        }
        double p_stay_o_next_origin = o_next_origin * p_next->getStayProb();
        p_stay_total += p_stay_o_next_origin;
        if ((p_stay_total - stay_th > MyUtils::DOUBLE_TOLENRANCE)
            || observation_dist.empty()) {
            // satisfy termination condition
            bool safe_policy = true;
            // STEP check safety



#ifdef SAFETY_CHECK
            for (int j = 0; j < observation_dist.size(); ++j) {
                if(enable_dp){
                    auto b = plan->getBelief(step);
                    if(belief_to_pi.find(b)!=belief_to_pi.end())
                    {
                        p_next = belief_to_pi[b];
                        if(!p_next)
                            safe_policy = false;

                        debug("safe policy found "<< safe_policy);
                        break;
                    }
                }
                BeliefStatePtr new_b_next = domain->getNextBelief(
                        plan->getBelief(step), a_next,
                        observation_dist[j].first, next_step);
                num_plan_checked += 1;
                if (new_b_next == nullptr) {
                    safe_policy = false;
//                    debug("unsafe policy found");
                    break;
                }

                if(enable_dp){

                    if(belief_to_pi.find(new_b_next)!=belief_to_pi.end())
                    {
                        p_next = belief_to_pi[new_b_next];
                        if(!p_next)
                            safe_policy = false;

                        debug("next safe policy found "<< safe_policy);
                        break;
                    }
                }
            }


#endif

            if (safe_policy) {
                policy->setStayProb(p_stay_total);
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "successfully complete the partial policy for action %s in step %d",
                          domain->getActionMeaning(a_next).c_str(), next_step
                );
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                break;
            }
#ifdef SAFETY_CHECK
            else {
                // fail to find a valid policy for this branch
                z3::expr new_constraints = plan->blockPlanPrefixConstraints(ctx, next_step);
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "fail to find a valid policy for the branch %s at step %d, the new constraints are:\n%s",
                          domain->getObservationMeaning(o_next).c_str(), next_step,
                          new_constraints.to_string().c_str());
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                return make_pair(nullptr, new_constraints);
            }
#endif
        }

        // STEP update upper bound
        double p_stay_o_next = o_next_prob * p_next->getStayProb();
        stay_th_temp = (stay_th_temp - p_stay_o_next) / (1 - o_next_prob);
        // STEP sample new observation
        o_next = MyUtils::sampleObservation(observation_dist);

            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "try to complete the partial policy for branch %s in step %d",
                      domain->getObservationMeaning(o_next).c_str(), next_step
            );
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
            // STEP get next belief as new initial belief
            BeliefStatePtr new_b_next = domain->getNextBelief(
                    plan->getBelief(step), a_next, o_next, next_step);
            num_plan_checked += 1;

            //STEP - MEMOIZATION OPTION FOR DYNAMIC PROGRAMMING
            if(enable_dp)
            {
                if(belief_to_pi.find(new_b_next)==belief_to_pi.end()){
                    if (new_b_next == nullptr) {
                        // unsafe new init
                        p_next = nullptr;
                    } else {
                        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                                 "new next belief for obervation branch in step %d - %s:\n%s",
                                 next_step,
                                 domain->getObservationMeaning(o_next).c_str(),
                                 domain->beliefStateToString(new_b_next, next_step).c_str()
                        );
                        MyUtils::printDebugMsg(msgBuffer,
                                               MyUtils::LEVEL_ONE_MSG);
                        POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(next_step, new_b_next, a_next, o_next);
                        if (MyUtils::solver_mode == "disable_bound_propagate") {
                            p_next = partialPolicySynthesis(new_domain,
                                                            stay_th, next_step, horizon_bound);
                        } else {
                            p_next = partialPolicySynthesis(new_domain,
                                                            stay_th_temp, next_step, horizon_bound);
                        }
                    }
                    //TODO keep track of p_next given o_next
                    belief_to_pi[new_b_next]= p_next;
                }
                else{
                    p_next = belief_to_pi[new_b_next];
//                    debug("reusing policy form map size "<<belief_to_pi.size());
                }
            }
            else
            {
                if (new_b_next == nullptr) {
                    // unsafe new init
                    p_next = nullptr;
                } else {
                    snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                             "new next belief for obervation branch in step %d - %s:\n%s",
                             next_step,
                             domain->getObservationMeaning(o_next).c_str(),
                             domain->beliefStateToString(new_b_next, next_step).c_str()
                    );
                    MyUtils::printDebugMsg(msgBuffer,
                                           MyUtils::LEVEL_ONE_MSG);
                    POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(next_step, new_b_next, a_next, o_next);
                    if (MyUtils::solver_mode == "disable_bound_propagate") {
                        p_next = partialPolicySynthesis(new_domain,
                                                        stay_th, next_step, horizon_bound);
                    } else {
                        p_next = partialPolicySynthesis(new_domain,
                                                        stay_th_temp, next_step, horizon_bound);
                    }
                }
            }


//#ifdef MEMOIZATION
//        if(belief_to_pi.find(new_b_next)==belief_to_pi.end()){
//            if (new_b_next == nullptr) {
//                // unsafe new init
//                p_next = nullptr;
//            } else {
//                snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
//                         "new next belief for obervation branch in step %d - %s:\n%s",
//                         next_step,
//                         domain->getObservationMeaning(o_next).c_str(),
//                         domain->beliefStateToString(new_b_next, next_step).c_str()
//                );
//                MyUtils::printDebugMsg(msgBuffer,
//                                       MyUtils::LEVEL_ONE_MSG);
//                POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(next_step, new_b_next, a_next, o_next);
//                if (MyUtils::solver_mode == "disable_bound_propagate") {
//                    p_next = partialPolicySynthesis(new_domain,
//                                                    stay_th, next_step, horizon_bound);
//                } else {
//                    p_next = partialPolicySynthesis(new_domain,
//                                                    stay_th_temp, next_step, horizon_bound);
//                }
//            }
//            //TODO keep track of p_next given o_next
//            belief_to_pi[new_b_next]= p_next;
//        }
//        else{
//            p_next = belief_to_pi[new_b_next];
//            debug("reusing policy form map size "<<belief_to_pi.size());
//        }
//#else
//        if (new_b_next == nullptr) {
//                // unsafe new init
//                p_next = nullptr;
//            } else {
//                snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
//                         "new next belief for obervation branch in step %d - %s:\n%s",
//                         next_step,
//                         domain->getObservationMeaning(o_next).c_str(),
//                         domain->beliefStateToString(new_b_next, next_step).c_str()
//                );
//                MyUtils::printDebugMsg(msgBuffer,
//                                       MyUtils::LEVEL_ONE_MSG);
//                POMDPDomainPtr new_domain = domain->createNewPOMDPDomain(next_step, new_b_next, a_next, o_next);
//                if (MyUtils::solver_mode == "disable_bound_propagate") {
//                    p_next = partialPolicySynthesis(new_domain,
//                                                    stay_th, next_step, horizon_bound);
//                } else {
//                    p_next = partialPolicySynthesis(new_domain,
//                                                    stay_th_temp, next_step, horizon_bound);
//                }
//            }
//#endif

        if (p_next == nullptr) {
            // fail to find a valid policy for this branch
            z3::expr new_constraints =
                    plan->blockPlanPrefixConstraints(ctx, next_step);
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "fail to find a valid policy for the branch %s at step %d, the new constraints are:\n%s",
                      domain->getObservationMeaning(o_next).c_str(), next_step,
                      new_constraints.to_string().c_str());
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
            return make_pair(nullptr, new_constraints);
        }
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "successfully complete the partial policy for branch %s in step: %d",
                  domain->getObservationMeaning(o_next).c_str(), next_step
        );

        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        policy->insertObservationBranch(o_next, p_next);
    }
    return make_pair(policy, ctx.bool_val(true));
}

#endif
