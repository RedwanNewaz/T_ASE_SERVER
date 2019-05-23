//
// Created by Yue Wang on 1/12/18.
//

#include "my_utils.h"
#include "partial_policy.h"
#include "pomdps/pomdp_domain.h"


#include <iostream>

PolicyNode::PolicyNode(BeliefStatePtr  b, int a_id,
                       POMDPDomainPtr d, int s, bool is_g):
        belief(b), child_nodes(),
        domain(d), step(s), is_goal(is_g) {
    action_idx = a_id;
    available_actions = domain->getAvailableActions();
    p_stay = 1.0;
}

void PolicyNode::insertObservationBranch(int o_id, PolicyNodePtr b_next) {
    child_nodes[o_id] = b_next;
}

pair<int, int> PolicyNode::getPolicyInfo(PolicyNodePtr policy) {
    if (policy == nullptr) {
        return make_pair(0, 0);
    }
    if (policy->getChildNodes().empty()) {
        // terminal node
        return make_pair(1, 1);
    }
    int max_child_depth = 0;
    int path_num = 0;
    for (auto child : policy->getChildNodes()) {
        auto child_info = getPolicyInfo(child.second);
        int child_depth = child_info.first;
        if (child_depth > max_child_depth) {
            max_child_depth = child_depth;
        }
        int child_path_num = child_info.second;
        path_num += child_path_num;
    }
    return make_pair(max_child_depth + 1, path_num);
}


void PolicyNode::printPolicy(PolicyNodePtr policy, stringstream &ss) {
    char msgBuffer[MyUtils::DEBUG_BUFFER_SIZE];
    if (policy == nullptr) {
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "empty policy tree:");
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
    }
    vector<PolicyNodePtr> policy_nodes;
    policy_nodes.push_back(policy);
    int step = 0;
    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "policy tree:");
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
    while (true) {
        ++ step;
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "step %d:",
                  step);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        vector<PolicyNodePtr> new_policy_nodes;
        for (auto policy_node : policy_nodes) {
            POMDPDomainPtr domain = policy_node->getDomain();
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "b_%d:\n%s\naction - %s\n observation distribution:",
                      step,
                      domain->beliefStateToString(policy_node->getBelief(), step - 1).c_str(),
                      domain->getActionMeaning(policy_node->getAction()).c_str());
            MyUtils::printMsgToString(msgBuffer, ss);
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
            vector<pair<int, double>> observation_dist =
                    domain->getObservationDistribution(policy_node->getAction(), policy_node->getBelief(), step);
            for (auto o_p: observation_dist) {
                ss << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") ";
                if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                    cout << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") ";
                }
            }
            ss << endl;
            ss << "next beliefs:" << endl;
            if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                cout << endl;
                cout << "next beliefs:" << endl;
            }
            for (auto child : policy_node->getChildNodes()) {
                POMDPDomainPtr domain = child.second->getDomain();
                snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                          "observation - %s:\n%s",
                          domain->getObservationMeaning(child.first).c_str(),
                          domain->beliefStateToString(child.second->getBelief(), step).c_str());
                MyUtils::printMsgToString(msgBuffer, ss);
                MyUtils::printDebugMsg(msgBuffer,
                                       MyUtils::LEVEL_ONE_MSG);
                if (child.second->isGoal()) {
                    ss << "reach goal" << endl;
                    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
                        cout << "reach goal" << endl;
                    }
                } else {
                    new_policy_nodes.push_back(child.second);
                }
            }
        }
        if (new_policy_nodes.empty()) {
            MyUtils::printMsgToString("terminated", ss);
            break;
        }
        policy_nodes = new_policy_nodes;
    }

}

double PolicyNode::calculateStayProb() {
    double stay_prob_computed = 0.0;
    if (child_nodes.empty()) {
        // terminal nodes
        stay_prob_computed = 1.0;
    } else {
        for (auto child : child_nodes) {
            int o_id = child.first;
            double observation_prob = 0.0;
            auto observation_dist = domain->getObservationDistribution(action_idx, belief, step);
            for (auto o_p : observation_dist) {
                if (o_p.first == o_id) {
                    observation_prob = o_p.second;
                }
            }
            if (observation_prob < MyUtils::DOUBLE_TOLENRANCE) {
                MyUtils::printErrorMsg("find a observation with 0 probability: "
                                       + std::to_string(o_id));
            }
            stay_prob_computed += observation_prob * child.second->calculateStayProb();
        }
    }
    /*if (getStayProb() - stay_prob_computed  > MyUtils::DOUBLE_TOLENRANCE) {
        MyUtils::printErrorMsg("stay probabiliy is not calculated correctly: "
                               + to_string(stay_prob_computed) + " != "
                               + to_string(getStayProb()));
    }*/
    setStayProb(stay_prob_computed);
    return stay_prob_computed;
}

void PolicyNode::policyEvaluation() {
    for (auto child : child_nodes) {
        PolicyNodePtr child_node = child.second;
        child_node->policyEvaluation();
    }
    domain->computeAlphaVector(shared_from_this());
}

void PolicyNode::collectAlphaVectors(vector<PolicyNodePtr> &alpha_vectors) {
    for (auto child : child_nodes) {
        PolicyNodePtr child_node = child.second;
        child_node->collectAlphaVectors(alpha_vectors);
    }
    alpha_vectors.push_back(shared_from_this());
}


void PolicyNode::writePolicyFile(PolicyNodePtr policy, ostream &out, int step){
    // sequential read policy and write it to a file
    if (!policy) {
        out << "\n";
    }
    else
    for (auto it=policy->child_nodes.begin(); it != policy->child_nodes.end(); ++it)
    {

        POMDPDomainPtr domain = policy->getDomain();

        vector<pair<int, double>> observation_dist =domain->getObservationDistribution(policy->getAction(), policy->getBelief(), step);

        double z;
        for (auto o_p: observation_dist) {
            z  = o_p.second;
//            out << "(" << domain->getObservationMeaning(o_p.first) << ", " << o_p.second << ") \n";
        }
        out<<"b"<<step-1<<","<<policy->getAction()<< ","<<z <<"\n" ;
        domain->publish(policy->getBelief(), step-1);
        writePolicyFile(it->second, out, step+1 );


    }

}



