#ifndef MY_UTILS_CC
#define MY_UTILS_CC

#include <iostream>

#include "belief_state.h"
#include "my_utils.h"


int MyUtils::debugLevel = 0;
const string MyUtils::OBSERVATION_VAR_PREFIX = "o";
const string MyUtils::ACTION_VAR_PREFIX = "a";
const string MyUtils::BELIEF_VAR_PREFIX = "b";
const string MyUtils::ROB_POS_VAR = "rob_pos";
const string MyUtils::STAY_IN_GOAL_ACTION = "reach goal and stay";
const double MyUtils::DOUBLE_TOLENRANCE = 0.000001;
const int MyUtils::DOUBLE_ROUND_FACTOR = 1000000;
string MyUtils::solver_mode = "";
string MyUtils::solver_name = "partial";

auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 MyUtils::generator(seed);
uniform_real_distribution<> MyUtils::uniDist{0, 1};

double MyUtils::uniform01() {
    return uniDist(generator);
}

double MyUtils::uniformReal(double lower_bound, double upper_bound) {
    return (upper_bound - lower_bound) * uniDist(generator) + lower_bound;
}

int MyUtils::uniformInt(int lower_bound, int upper_bound) {
    auto r = (int)floor(uniformReal((double)lower_bound, (double)(upper_bound) + 1.0));
    return (r > upper_bound) ? upper_bound : r;
}


int MyUtils::sampleObservation(const vector<pair<int, double>> &observation_dist) {
    double rand_point = MyUtils::uniform01();
    int obs_idx = -1;
    for (int i = 0; i < observation_dist.size(); ++i) {
        double obs_prob = observation_dist[i].second;
        if (obs_prob < DOUBLE_TOLENRANCE) {
            continue;
        }
        if (rand_point < obs_prob) {
            obs_idx = i;
            break;
        }
        rand_point -= obs_prob;
    }
    if (obs_idx == -1) {
        MyUtils::printErrorMsg("no observation can be sampled");
    }
    return observation_dist[obs_idx].first;
}

double MyUtils::getAlphaForState(unordered_map<int, double> alpha, int idx) {
    if (alpha.find(idx) != alpha.end()) {
        return alpha[idx];
    }
    return 0.0;
}

void MyUtils::setDebugLevel (int debug_level) {
	debugLevel = debug_level;
}

void MyUtils::printDebugMsg (string msg, int debug_level) {
	if (debug_level <= debugLevel) {
		cout << msg << endl;
	}
}

void MyUtils::printMsgToString(string msg, stringstream &ss) {
	ss << msg << endl;
}

void MyUtils::printErrorMsg (string msg) {
	cout << "Error: " << msg << endl;
	exit (EXIT_FAILURE);
}

void MyUtils::printZ3Expr(string msg, const z3::expr &exp, int debug_level) {
    if (debug_level <= debugLevel) {
        cout << msg << endl;
        cout << exp << endl;
    }
}

point MyUtils::now() {
	return chrono::system_clock::now();
}

double MyUtils::run_time(const duration &d)
{
	return chrono::duration<double>(d).count();
}

bool MyUtils::checkBeliefEquality(BeliefStatePtr b1, BeliefStatePtr b2) {
    auto b1_vals = b1->getBeliefStateValues();
    auto b2_vals = b2->getBeliefStateValues();
    if (b1_vals.size() != b2_vals.size()) {
        return false;
    }
    for (int i = 0; i < b1_vals.size(); ++i) {
        if (abs(b1_vals[i] - b2_vals[i]) > 0.001) {
            return false;
        }
    }
    return true;
}

#endif
