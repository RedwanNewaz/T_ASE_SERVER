//
// Created by Yue Wang on 9/24/18.
//

#include "solvers/quantitative_solver.h"

PolicyNodePtr QuantitativeSolver::solve(POMDPDomainPtr domain, int start_step, int max_k, bool run) {

    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "Value threshold : %f", POMDPDomain::epsilon);
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);

    point start = MyUtils::now();
    PolicyNodePtr policy = quantitativePolicySynthesis(domain, stay_threshold, start_step, max_k);
    point end = MyUtils::now();
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "partial policy generation time : %fs", MyUtils::run_time(end - start));
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    one_run_planning_num = 1;
    if (policy != nullptr && run) {
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

PolicyNodePtr QuantitativeSolver::quantitativePolicySynthesis(POMDPDomainPtr domain,
                                                         double stay_th, int start_step, int horizon_bound) {
    point start = MyUtils::now();
    PolicyNodePtr policy = partialPolicySynthesis(domain, stay_th, start_step, horizon_bound);
    point end = MyUtils::now();
    double synthesis_time =  MyUtils::run_time(end - start);
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "policy construction time: " << synthesis_time << "s" << endl;
    }


    if (policy == nullptr) {
        return nullptr;
    }
    vector<PolicyNodePtr> alpha_vectors_pre;
    policy->policyEvaluation();
    policy->collectAlphaVectors(alpha_vectors_pre);
    double policy_value = policy->getDomain()->computeBeliefValue(
            policy->getBelief(), policy->getAlphaVector(), start_step);

    while (true) {
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "policy value: " << policy_value << endl;
            cout << "alpha vector size: " << alpha_vectors_pre.size() << endl;
        }
        vector<PolicyNodePtr> alpha_vectors_new;
        point start = MyUtils::now();

        double improve = 0.0;
        for (int i = 0; i < alpha_vectors_pre.size(); ++i) {
            PolicyNodePtr node = alpha_vectors_pre[i];
            improve += policyImprovement(alpha_vectors_pre, alpha_vectors_new, node, horizon_bound);
        }

        point end = MyUtils::now();
        double one_iter_time =  MyUtils::run_time(end - start);
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "one iteration time: " << one_iter_time << "s" << endl;
        }
        policy = alpha_vectors_new[alpha_vectors_new.size() - 1];

        if (improve / alpha_vectors_pre.size() < POMDPDomain::epsilon) {
            break;
        }

        policy_value = policy->getDomain()->computeBeliefValue(
                policy->getBelief(), policy->getAlphaVector(), start_step);

        // filter duplicate nodes
        unordered_map<string, int> alpha_vector_map;
        unordered_map<int, bool> use;
        for (int i = 0; i < alpha_vectors_new.size(); ++i) {
            use[i] = false;
            PolicyNodePtr node = alpha_vectors_new[i];
            BeliefStatePtr b = node->getBelief();
            string b_str = node->getDomain()->beliefStateToString(b, node->getStep() - 1);
            if (alpha_vector_map.find(b_str) == alpha_vector_map.end()) {
                alpha_vector_map[b_str] = i;
            } else {
                PolicyNodePtr node_old = alpha_vectors_new[alpha_vector_map[b_str]];
                double value_old = node_old->getDomain()->computeBeliefValue(node_old->getBelief(), node_old->getAlphaVector(), node_old->getStep() - 1);
                double value_new = node->getDomain()->computeBeliefValue(b, node->getAlphaVector(), node->getStep() - 1);
                if (value_new > value_old) {
                    alpha_vector_map[b_str] = i;
                }
            }
        }

        vector<PolicyNodePtr> alpha_vectors;
        for (auto alpha_vector : alpha_vector_map) {
            use[alpha_vector.second] = true;
        }
        for (int i = 0; i < alpha_vectors_new.size(); ++i) {
            if (use[i]) {
                alpha_vectors.push_back(alpha_vectors_new[i]);
            }
        }
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "before pruning, alpha vector size: " << alpha_vectors_new.size() << endl;
            cout << "after pruning, alpha vector size: " << alpha_vectors.size() << endl;
            cout << "average value difference: " << improve / alpha_vectors_pre.size() << endl;
        }
        alpha_vectors_pre = alpha_vectors;

    }
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "optimal policy value: " << policy->getDomain()->computeBeliefValue(
                policy->getBelief(), policy->getAlphaVector(), start_step) << endl;
        cout << "optimal policy alpha vector size: " << alpha_vectors_pre.size() << endl;
    }
    return policy;
}

double QuantitativeSolver::policyImprovement(const vector<PolicyNodePtr> &alpha_vectors_pre,
                                        vector<PolicyNodePtr> &alpha_vectors_new,
                                        PolicyNodePtr policy, int horizon_bound) {
    if (policy->isGoal()) {
        // terminal belief, no need to improve
        alpha_vectors_new.push_back(policy);
        return 0;
    }
    int step = policy->getStep() - 1;
    double previous_value = policy->getDomain()->computeBeliefValue(policy->getBelief(),
                                                                    policy->getAlphaVector(), step);

    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << endl << "try to improve action "
             <<  policy->getDomain()->getActionMeaning(policy->getAction())
             << " at step: " << policy->getStep() << endl;
    }
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "previous value: " << previous_value << endl;
    }
    PolicyNodePtr best_alpha = policy->getDomain()->backup(alpha_vectors_pre, alpha_vectors_new, policy,
                                                           shared_from_this(), step, horizon_bound);
    double new_value = best_alpha->getDomain()->computeBeliefValue(
            best_alpha->getBelief(), best_alpha->getAlphaVector(), step);

    if (new_value < previous_value + POMDPDomain::epsilon) {
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "action " << policy->getDomain()->getActionMeaning(policy->getAction())
                 << " at step " << policy->getStep()  << " is already optimal" << endl;
        }
        alpha_vectors_new.push_back(policy);
        return 0;
    }

    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "value improved: " << new_value << endl;
    }
    if (best_alpha->getAction() == policy->getAction()) {
        // update policy for further improvement
        for (auto child : best_alpha->getChildNodes()) {
            int obs_id = child.first;
            PolicyNodePtr child_node = child.second;
            policy->insertObservationBranch(obs_id, child_node);
        }
        policy->getDomain()->computeAlphaVector(policy);
        alpha_vectors_new.push_back(best_alpha);
    } else {
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "find a better action choice " << best_alpha->getDomain()->getActionMeaning(best_alpha->getAction())
                 << " for step " << policy->getStep() << endl;
        }
        best_alpha->collectAlphaVectors(alpha_vectors_new);
    }
    return new_value - previous_value;
}

