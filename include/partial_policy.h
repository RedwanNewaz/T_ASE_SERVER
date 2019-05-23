//
// Created by Yue Wang on 1/12/18.
//

#ifndef POLICY_SYNTHESIZER_PARTIAL_POLICY_H
#define POLICY_SYNTHESIZER_PARTIAL_POLICY_H


#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <z3++.h>

#include "belief_state.h"


using namespace std;

class POMDPDomain;
typedef shared_ptr <POMDPDomain> POMDPDomainPtr;

class PolicyNode;
typedef shared_ptr <PolicyNode> PolicyNodePtr;

class PolicyNode
        : public enable_shared_from_this<PolicyNode> {
public:
    PolicyNode(BeliefStatePtr b, int a_id,
                POMDPDomainPtr d, int s, bool is_g = false);

    BeliefStatePtr getBelief() {
        return belief;
    }
    void insertObservationBranch(int o_id, PolicyNodePtr b_next);

    void setStayProb (double stay_prob) {
        p_stay = stay_prob;
    }
    double getStayProb () {
        return p_stay;
    }

    unordered_map<int, bool> &getAvailableActions() {
        return available_actions;
    };

    void setAvailableActions(const unordered_map<int, bool> &actions) {
        available_actions = actions;
    }

    int getAction() {
        return action_idx;
    }

    int getStep() {
        return step;
    }

    double getAlpha(int idx) {
        if (alpha.find(idx) == alpha.end()) {
            return 0.0;
        }
        return alpha[idx];
    }

    unordered_map<int, double> &getAlphaVector() {
        return alpha;
    };

    void setAlpha(unordered_map<int, double> &a) {
        alpha = a;
    }

    void policyEvaluation();

    void collectAlphaVectors(vector<PolicyNodePtr> &alpha_vectors);

    unordered_map<int, PolicyNodePtr> &getChildNodes() {
        return child_nodes;
    }

    bool isGoal() {
        return is_goal;
    }

    POMDPDomainPtr getDomain() {
        return domain;
    }

    static void printPolicy(PolicyNodePtr policy,
                            stringstream &ss);
    double calculateStayProb();
    static pair<int, int> getPolicyInfo(PolicyNodePtr policy);

    void writePolicyFile(PolicyNodePtr policy, ostream &out, int step =1);

private:
    unordered_map<int, PolicyNodePtr> child_nodes;
    int action_idx;
    BeliefStatePtr belief;
    double p_stay;
    bool is_goal;
    POMDPDomainPtr domain;
    unordered_map<int, double> alpha;
    unordered_map<int, bool> available_actions;
    int step;
};

#endif //POLICY_SYNTHESIZER_PARTIAL_POLICY_H
