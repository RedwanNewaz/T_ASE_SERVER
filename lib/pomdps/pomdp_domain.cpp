//
// Created by Yue Wang on 1/20/18.
//

#include "pomdps/pomdp_domain.h"
#include "plan.h"
#include "solvers/solver.h"
#include "solvers/partial_solver.h"

double POMDPDomain::epsilon = 0.0;

POMDPDomain::POMDPDomain(int s, int a_s, int o_s){
    start_step = s;
    a_start = a_s;
    o_start = o_s;
}

BeliefStatePtr POMDPDomain::getNextBelief(BeliefStatePtr b_pre, int action_idx,
                                          int observation_idx, int step) {
    z3::context ctx;
    z3::solver solver(ctx);
    z3::expr_vector b_pre_conds(ctx);
    vector<z3::expr> b_pre_vars = getBeliefStateVarList(ctx, step - 1);
    vector<z3::expr> b_pre_vals = b_pre->getBeliefStateExprs(ctx);
    assert(b_pre_vars.size() == b_pre_vals.size()&& "when generating next belief, the sizes of belief vars and belief values do not agree");

    string a_i_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_i = ctx.int_const(a_i_name.c_str());

    string o_i_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr o_i = ctx.int_const(o_i_name.c_str());
    for (int i = 0; i < b_pre_vals.size(); ++i) {
        b_pre_conds.push_back(b_pre_vars[i] == b_pre_vals[i]);
    }


    solver.add(mk_and(b_pre_conds));
    solver.add(genTransCond(ctx, step)); // generate transition condition
    solver.add(a_i == ctx.int_val(action_idx));
    solver.add(o_i == ctx.int_val(observation_idx));
    z3::check_result ret = solver.check();

    if (ret == z3::sat) {
        ModelPtr model = make_shared<z3::model>(solver.get_model());
        return getBeliefState(ctx, model, step);
    }
    return nullptr;
}

PolicyNodePtr POMDPDomain::checkPolicy(BeliefStatePtr b, PolicyNodePtr policy, int step, int horizon_bound) {
    if (step > horizon_bound) {
        return nullptr;
    }
    int next_step = step + 1;
    if (policy->isGoal()) {
        if (isGoal(b, step)) {
            return make_shared <PolicyNode> (b, 0, shared_from_this(), step + 1, true);
        }
        return nullptr;
    }
    int action_id = policy->getAction();
    PolicyNodePtr new_policy = make_shared <PolicyNode> (b, action_id, shared_from_this(), next_step);
    for (auto child : policy->getChildNodes()) {
        int obs_id = child.first;
        PolicyNodePtr child_node = child.second;
        BeliefStatePtr b_a_o = getNextBeliefDirect(b, action_id, obs_id, next_step);
        if (check) {
            BeliefStatePtr b_a_o_z3 = getNextBelief(b, action_id, obs_id, next_step);
            if (b_a_o == nullptr) {
                if (b_a_o_z3 != nullptr) {
                    cout << "somthing is wrong with belief update" << endl;
                    cout << "previous belief: " << beliefStateToString(b, step) << endl;
                    cout << "action: " << getActionMeaning(action_id) << ", observation: "
                         << getObservationMeaning(obs_id) << endl;
                    cout << "from update: invalid" << endl;
                    cout << "from z3: " << beliefStateToString(b_a_o_z3, next_step) << endl;
                    exit (EXIT_FAILURE);
                }
            } else {
                if (b_a_o_z3 == nullptr) {
                    cout << "somthing is wrong with belief update" << endl;
                    cout << "previous belief: " << beliefStateToString(b, step) << endl;
                    cout << "action: " << getActionMeaning(action_id) << ", observation: "
                         << getObservationMeaning(obs_id) << endl;
                    cout << "from update: " << beliefStateToString(b_a_o, next_step) << endl;
                    cout << "from z3: invalid" << endl;
                    exit (EXIT_FAILURE);
                }
                if (!MyUtils::checkBeliefEquality(b_a_o, b_a_o_z3)) {
                    cout << "somthing is wrong with belief update" << endl;
                    cout << "previous belief: " << beliefStateToString(b, step) << endl;
                    cout << "action: " << getActionMeaning(action_id) << ", observation: "
                         << getObservationMeaning(obs_id) << endl;

                    cout << "from update: " << beliefStateToString(b_a_o, next_step) << endl;
                    auto b_a_o_vals = b_a_o->getBeliefStateValues();
                    for (int i = 0; i < b_a_o_vals.size(); ++i) {
                        cout << b_a_o_vals[i] << ", ";
                    }
                    cout << endl;
                    cout << "from z3: " << beliefStateToString(b_a_o_z3, next_step) << endl;
                    auto b_a_o_z3_vals = b_a_o_z3->getBeliefStateValues();
                    for (int i = 0; i < b_a_o_z3_vals.size(); ++i) {
                        cout << b_a_o_z3_vals[i] << ", ";
                    }
                    cout << endl;
                    exit (EXIT_FAILURE);
                }
            }
        }
        if (b_a_o == nullptr) {
            return nullptr;
        }

        PolicyNodePtr new_child_node = checkPolicy(b_a_o, child_node, next_step, horizon_bound);
        if (new_child_node == nullptr) {
            return nullptr;
        }
        // new_child_node->setAvailableActions(child_node->getAvailableActions());
        new_policy->insertObservationBranch(obs_id, new_child_node);
    }
    return new_policy;
}


PolicyNodePtr POMDPDomain::backup(const vector<PolicyNodePtr> &alpha_vectors_pre,
                                   vector<PolicyNodePtr> &alpha_vectors_new,
                                   PolicyNodePtr policy, PolicySolverPtr solver, int step, int horizon_bound) {
    BeliefStatePtr b = policy->getBelief();
    int next_step = step + 1;

    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "current belief: ";
        cout << beliefStateToString(b, step) << endl;
        cout << "previous action choice at step "
             << next_step << ": " << getActionMeaning(policy->getAction())
             << ", value: " << computeBeliefValue(b, policy->getAlphaVector(), step)
             << endl;
    }

    double best_a_value = 0.0;
    PolicyNodePtr best_alpha = nullptr;
    auto &available_actions = policy->getAvailableActions();

    vector<PolicyNodePtr> alpha_vectors = alpha_vectors_pre;
    for (auto child : policy->getChildNodes()) {
        PolicyNodePtr child_node = child.second;
        alpha_vectors.push_back(child_node);
        PolicyNodePtr p_a_o = child_node->getDomain()->checkPolicy(child_node->getBelief(), child_node, next_step, horizon_bound);
        if (p_a_o == nullptr) {
            MyUtils::printDebugMsg("Warning: something is wrong", MyUtils::LEVEL_ZERO_MSG);
        }
    }
    for (auto &action_info : available_actions) {

        bool available = action_info.second;
        if (!available) {
            continue;
        }
        int action_id = action_info.first;
        MyUtils::printDebugMsg("compute alpha vector for action: "
                               + getActionMeaning(action_id),  MyUtils::LEVEL_ONE_MSG);
        auto obs_dist = getObservationDistribution(action_id, b, next_step);
        unordered_map<int, PolicyNodePtr> best_alpha_node;
        bool fail = false;
        for (auto obs : obs_dist) {
            int obs_id = obs.first;
            BeliefStatePtr b_a_o = getNextBeliefDirect(b, action_id, obs_id, next_step);
            if (b_a_o == nullptr) {
                fail = true;
                break;
            }
            double best_v = 0;
            PolicyNodePtr best_alpha_a_o = nullptr;
            for (int i = 0; i < alpha_vectors.size(); ++i) {
                PolicyNodePtr node = alpha_vectors[i];

                double value_estimated = computeBeliefValue(b_a_o, node->getAlphaVector(), next_step);
                if (value_estimated > best_v || best_alpha_a_o == nullptr) {
                    PolicyNodePtr p_a_o = checkPolicy(b_a_o, node, next_step, horizon_bound);
                    if (p_a_o == nullptr) {
                        continue;
                    }
                    p_a_o->policyEvaluation();
                    double v_alpha_a_o = computeBeliefValue(b_a_o, p_a_o->getAlphaVector(), next_step);
                    if (v_alpha_a_o > best_v || best_alpha_a_o == nullptr) {
                        best_alpha_a_o = p_a_o;
                        best_v = v_alpha_a_o;
                    }
                }
            }
            if (best_alpha_a_o == nullptr) {
                if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                    cout << "No alpha vector is valid for observation: " << getObservationMeaning(obs_id) << endl;
                }
                if (MyUtils::solver_mode == "nonexplore") {
                    fail = true;
                    break;
                } else {
                    POMDPDomainPtr new_domain = createNewPOMDPDomain(next_step, b_a_o, action_id, obs_id);

                    point start = MyUtils::now();
                    best_alpha_a_o = std::static_pointer_cast<PartialSolver>(solver)->partialPolicySynthesis(new_domain,
                                                                                                             std::static_pointer_cast<PartialSolver>(solver)->stay_threshold, next_step, horizon_bound);
                    point end = MyUtils::now();
                    double synthesis_time =  MyUtils::run_time(end - start);
                    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                        cout << "exploration time: " << synthesis_time << "s" << endl;
                    }

                    if (best_alpha_a_o == nullptr) {
                        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                            cout << "can not construct a new policy for observation: " << getObservationMeaning(obs_id) << endl;
                        }
                        fail = true;
                        break;
                    }
                    best_alpha_a_o->policyEvaluation();
                    best_alpha_a_o->collectAlphaVectors(alpha_vectors_new);
                }
            }
            best_alpha_node[obs_id] = best_alpha_a_o;
            if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                if (best_alpha_a_o->isGoal()) {
                    cout << "best action for observation " << getObservationMeaning(obs_id) << ": "
                         << "reach goal" << ", value: " << best_v << endl;
                } else {
                    cout << "best action for observation " << getObservationMeaning(obs_id) << ": "
                         << getActionMeaning(best_alpha_a_o->getAction()) << ", value: " << best_v << endl;
                }
            }
        }
        if (fail) {
            MyUtils::printDebugMsg("action: "
                                   + getActionMeaning(action_id)
                                   + " is not valid",  MyUtils::LEVEL_ONE_MSG);
            available_actions[action_id] = false;
            continue;
        }

        PolicyNodePtr p_a = make_shared <PolicyNode> (b, action_id, shared_from_this(), next_step);
        for (auto alpha_a_o : best_alpha_node) {
            int obs_id = alpha_a_o.first;
            PolicyNodePtr child = alpha_a_o.second;
            p_a->insertObservationBranch(obs_id, child);
        }
        computeAlphaVector(p_a);
        double value_a = computeBeliefValue(b, p_a->getAlphaVector(), step);
        MyUtils::printDebugMsg("value for action "
                               + getActionMeaning(action_id) + ": "
                               + to_string(value_a),  MyUtils::LEVEL_ONE_MSG);


        if (value_a > best_a_value || best_alpha == nullptr) {
            best_a_value = value_a;
            if ((best_alpha != nullptr) && (MyUtils::solver_mode == "exact")) {
                best_alpha->collectAlphaVectors(alpha_vectors_new);
            }
            best_alpha = p_a;
        } else if (MyUtils::solver_mode == "exact") {
            p_a->collectAlphaVectors(alpha_vectors_new);
        }
    }
    if (best_alpha == nullptr) {
        MyUtils::printDebugMsg("Warning: no valid action", MyUtils::LEVEL_ZERO_MSG);
        return policy;
    }

    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "best action choice at step "
             << step + 1 << ": " << getActionMeaning(best_alpha->getAction())
             << ", value: " << best_a_value
             << endl;
    }
    best_alpha->setAvailableActions(available_actions);
    return best_alpha;
}


