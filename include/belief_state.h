//
// Created by Yue Wang on 4/5/18.
//

#ifndef POLICY_SYNTHESIZER_BELIEF_STATE_H
#define POLICY_SYNTHESIZER_BELIEF_STATE_H

#include <vector>
#include <z3++.h>
#include "my_utils.h"

using namespace std;

class BeliefState;
typedef shared_ptr <BeliefState> BeliefStatePtr;

class BeliefState {
public:
    virtual ~BeliefState();

    BeliefState(const vector<z3::expr> &b);
    BeliefState(const vector<pair<double, double>> &b);
    BeliefState(const vector<double> &b);
    vector<double> getBeliefStateValues();
    vector<z3::expr> getBeliefStateExprs(z3::context &ctx);
    const vector<pair<double, double>> &getBeliefStateExactValues() {
        return belief_state;
    };
private:

    vector<pair<double, double>> belief_state;
};

#endif //POLICY_SYNTHESIZER_BELIEF_STATE_H
