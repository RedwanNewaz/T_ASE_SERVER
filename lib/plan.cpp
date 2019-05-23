//
// Created by Yue Wang on 1/12/18.
//
#ifndef PLAN_CC
#define PLAN_CC

#include "plan.h"
#include "pomdps/pomdp_domain.h"


Plan::Plan(z3::context &ctx, ModelPtr model, POMDPDomainPtr domain, int s, int h) {
    start_step = s;
    horizon_bound = h;
    // construct the plan from the model
    BeliefStatePtr b_pre = domain->getBeliefState(ctx, model, start_step);
    belief_list.push_back(b_pre);
    for (int i = start_step + 1; i <= horizon_bound; ++i) {
        string a_i_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(i);
        z3::expr a_i = ctx.int_const(a_i_name.c_str());
        z3::expr a_i_val = model->eval(a_i);
        int action_id = a_i_val.get_numeral_int();
        action_list.push_back(action_id);

        string o_i_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(i);
        z3::expr o_i = ctx.int_const(o_i_name.c_str());
        z3::expr o_i_val = model->eval(o_i);
        int observation_id = o_i_val.get_numeral_int();

        observation_list.push_back(observation_id);

        observation_dist_list.push_back(
                domain->getObservationDistribution(action_id, b_pre, i));
        b_pre = domain->getBeliefState(ctx, model, i);
        belief_list.push_back(b_pre);
    }

    MyUtils::printDebugMsg("plan size "+to_string(belief_list.size()), MyUtils::LEVEL_TWO_MSG);
}

int Plan::getAction(unsigned step) {
    int idx = step - start_step - 1;
    return action_list[idx];
}

int  Plan::getObservation(unsigned step) {
    int idx = step - start_step - 1;
    return observation_list[idx];
}

const vector<pair<int, double>> &Plan::getObservationDist(unsigned step) {
    int idx = step - start_step - 1;
    return observation_dist_list[idx];
};

BeliefStatePtr Plan::getBelief(unsigned step) {
    int idx = step - start_step;
    return belief_list[idx];
}

z3::expr Plan::blockPlanPrefixConstraints(z3::context &ctx, int prefix_bound) {
    z3::expr_vector block_constraints (ctx);
    for (int i = start_step + 1; i < prefix_bound; ++i) {
        string a_i_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(i);
        z3::expr a_i = ctx.int_const(a_i_name.c_str());
        z3::expr a_i_val = ctx.int_val(getAction(i));
        block_constraints.push_back(a_i == a_i_val);

        string o_i_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(i);
        z3::expr o_i = ctx.int_const(o_i_name.c_str());
        z3::expr o_i_val = ctx.int_val(getObservation(i));
        block_constraints.push_back(o_i == o_i_val);
    }
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(prefix_bound);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    z3::expr a_k_val = ctx.int_val(getAction(prefix_bound));
    block_constraints.push_back(a_k == a_k_val);
    return !mk_and(block_constraints);
}

void Plan::printPlan(POMDPDomainPtr domain) {
    char msgBuffer[MyUtils::DEBUG_BUFFER_SIZE];
    BeliefStatePtr b_0 = getBelief(start_step);
    snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
             "initial belief in step %d:\n%s",
             start_step,
             domain->beliefStateToString(b_0, start_step).c_str());
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    for (int i = start_step + 1; i <= horizon_bound; ++i) {
        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                 "step %d:", i);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);

        int a_i_val = getAction(i);
        int o_i_val = getObservation(i);

        // TODO visualization

        const vector<pair<int, double>> &observation_dist = getObservationDist(i);
        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                 "observation distribution:");
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        for (auto o_p: observation_dist) {
            if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                cout << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") ";
            }
        }
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << endl;
        }
        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                 "%s: %s", domain->getActionMeaning(a_i_val).c_str(),
                 domain->getObservationMeaning(o_i_val).c_str());
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        BeliefStatePtr b_i = getBelief(i);
        snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                 "next belief:\n%s", domain->beliefStateToString(b_i, i).c_str());
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
    }
}

#endif
