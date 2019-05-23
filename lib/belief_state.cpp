//
// Created by Yue Wang on 4/5/18.
//

#include "belief_state.h"

BeliefState::BeliefState(const vector<z3::expr> &b): belief_state() {
    for (auto &b_i : b) {
        z3::expr n = b_i.numerator();
        z3::expr d = b_i.denominator();
        belief_state.push_back(make_pair(
                stod(n.get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str()),
                stod(d.get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str())));
    }
}

BeliefState::BeliefState(const vector<pair<double, double>> &b): belief_state(b) {
}

BeliefState::BeliefState(const vector<double> &b): belief_state() {
    for (int i = 0; i < b.size(); ++i) {
        belief_state.push_back(make_pair(int(b[i] * MyUtils::DOUBLE_ROUND_FACTOR), MyUtils::DOUBLE_ROUND_FACTOR));
    }
}

vector<double> BeliefState::getBeliefStateValues() {
    vector<double> b_vals;
    for (auto &b_i : belief_state) {
        b_vals.push_back(b_i.first / b_i.second);
    }
    return b_vals;
}

vector<z3::expr> BeliefState::getBeliefStateExprs(z3::context &ctx) {
    vector<z3::expr> b_exprs;
    for (auto &b_i : belief_state) {
        b_exprs.push_back((ctx.real_val(std::to_string(b_i.first).c_str())
                           / ctx.real_val(std::to_string(b_i.second).c_str())));
    }
    return b_exprs;
}

BeliefState::~BeliefState() {
    belief_state.clear();
}
