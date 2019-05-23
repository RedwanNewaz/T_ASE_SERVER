//
// Created by Redwan Newaz on 2019-04-27.
//

#include "kitchen/kitchen_belief.h"
//#include "kitchen/kitchen_tau.h"
#include "my_utils.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_set>



using namespace std;

int kitchen_belief::x_size = 0;
int kitchen_belief::y_size = 0;
int kitchen_belief::rob_locs_size = 0;
int kitchen_belief::ready_robot_loc = 0;
int kitchen_belief::unsafe_robot_loc = 0;
int kitchen_belief::goal_robot_loc = 0;
int kitchen_belief::num_obs = 0;
double kitchen_belief::delta = 0.0;
vector<int> kitchen_belief::obs_locs;

double kitchen_belief::p_fp = 0.02; // observation
double kitchen_belief::p_fn = 0.05; // observation
double kitchen_belief::p_succ = 0.98; //# transition probability

kitchen_belief::kitchen_belief(int s, BeliefStatePtr init_b,
                               int a_s, int o_s,
                               unordered_map<int, int> init_b_to_g_map, bool run_init):
        POMDPDomain(s, a_s, o_s), obs_all_locs(), obs_locs_list(),
        belief_group_size(), belief_to_group_maps() {

    init(run_init);

    const vector<pair<double, double>> &init_b_vals
            = init_b->getBeliefStateExactValues();
    vector<pair<double, double>> new_init_b_vals;
    unordered_map<int, int> group_remap;
    int group_num = 0;
    for (int g = 0; g < init_b_vals.size(); ++g) {
        double n = init_b_vals[g].first;
        double d = init_b_vals[g].second;
        if (n == 0) {
            continue;
        }
        group_remap[g] = group_num;
        ++ group_num;
        new_init_b_vals.push_back(make_pair(n, d));
        belief_group_size[start_step].push_back(0);
    }

    for (auto b_to_g : init_b_to_g_map) {
        int idx = b_to_g.first;
        int g_idx = b_to_g.second;

        if (group_remap.find(g_idx) != group_remap.end()) {
            int new_g_idx = group_remap[g_idx];
            belief_to_group_maps[start_step][idx] = new_g_idx;
            ++belief_group_size[start_step][new_g_idx];
        }
    }

    init_belief = make_shared <BeliefState> (new_init_b_vals);
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << beliefStateToString(init_belief, start_step) << endl;
    }

    see_obs_num = 0;
    init_robot_loc = -1;
    auto probs = extractProbs(init_belief, start_step);
    auto &b_r_vals = probs.first;
    double max_rob_prob = 0.0;
    for (int r = 0; r < rob_locs_size; ++r) {
        if (b_r_vals[r] > max_rob_prob) {
            max_rob_prob = b_r_vals[r];
            init_robot_loc = r;
        }
    }

    MyUtils::printDebugMsg("init robot loc: " + std::to_string(init_robot_loc),
                           MyUtils::LEVEL_ONE_MSG);

    auto &b_obs_vals = probs.second;
    vector<double> obs_probs;
    for (int i = 0; i < obs_all_locs.size(); ++i) {
        obs_probs.push_back(0.0);
    }
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        double obs_i_prob = b_obs_vals[i];
        for (auto obs_loc : obs_locs_list[i]) {
            int idx = obs_loc - y_size;
            obs_probs[idx] += obs_i_prob;
        }
    }
    for (int i = 0; i < obs_all_locs.size(); ++i) {
        //STEP - observation threshold 0.6
        if (obs_probs[i] > 0.6) {
            ++ see_obs_num;
        }
    }
}



BeliefStatePtr kitchen_belief::filterZeroProb(BeliefStatePtr b, int step) {

    // filer zero probability
    auto probs = extractProbs(b, step);
    auto &b_r_vals = probs.first;
    vector<bool> is_zero;
    for (int r = 0; r < rob_locs_size; ++r) {
        if (b_r_vals[r] < MyUtils::DOUBLE_TOLENRANCE) {
            is_zero.push_back(true);
        } else {
            is_zero.push_back(false);
        }
    }
    vector<pair<double, double>> new_b_vals
            = b->getBeliefStateExactValues();

    for (auto b_to_g : belief_to_group_maps[step]) {
        int idx = b_to_g.first;
        int r = getRobotLocIdx(idx);
        int g_idx = b_to_g.second;

        if (is_zero[r]) {
            new_b_vals[g_idx].first = 0.0;
            new_b_vals[g_idx].second = 1.0;
        }
    }
    return make_shared <BeliefState> (new_b_vals);
}

void kitchen_belief::genObstacleLocsList(int obs_idx, vector<int> &obs_locs_temp, int loc_idx,
                                         const vector<int> &obs_all_locs,
                                         vector<vector<int>> &obs_locs_list) {

/**@brief
 * Get all possible combination of obstacles \par
 * this function is reimplemented with while loop to avoid recursion. \par
 * Resulting significant performance improvement
 * @param obs_all_locs
 * @return obs_locs_list
 */

    auto nCr = [&](int n, int k){
        if (k > n) {
            return 0;
        }
        int r = 1;
        for (int d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    };

    int n = obs_all_locs.size();
    int r = num_obs;
    int num_states = nCr(n,r);
    obs_locs_list.resize(num_states);

    std::vector<bool> v(n);
    std::fill(v.end() - r, v.end(), true);
    int count = 0;
    do {
        vector<int> rowVal;
        for (int i = 0; i < n; ++i)
            if (v[i])
                rowVal.emplace_back(obs_all_locs[i]); // TODO: need to change?
        obs_locs_list[count]=rowVal;
        ++count;
    } while (std::next_permutation(v.begin(), v.end()));

    /*
     obs = np.arange(12, 26)
     comb = combinations(obs, num_obs)
     return list(comb)
     */
//    if (obs_idx == num_obs) {
//        obs_locs_list.push_back(std::vector<int> (obs_locs_temp));
//    } else {
//        for (int i = loc_idx; i < obs_all_locs.size(); ++i) {
//            obs_locs_temp.push_back(obs_all_locs[i]);
//            genObstacleLocsList(obs_idx + 1, obs_locs_temp, i + 1, obs_all_locs, obs_locs_list);
//            obs_locs_temp.pop_back();
//        }
////        cout<< "obs_loc_list size " <<obs_locs_list.size()<<endl;
//    }
}


int kitchen_belief::getNextLoc(int action_idx, int rob_loc_pre) {
    if (rob_loc_pre == ready_robot_loc || rob_loc_pre == unsafe_robot_loc
        || rob_loc_pre == goal_robot_loc) {
        return rob_loc_pre;
    }
    auto xy = getXY(rob_loc_pre);
    int x = xy.first;
    int y = xy.second;
    int next_rob_loc = -1;

    int north_bound = num_obs;
    if (num_obs < 3) {
        north_bound = 3;
    }
    int move_north_bound = north_bound + 1;
    if (MyUtils::solver_mode == "success") {
        move_north_bound = north_bound + 2;
    }
    switch (action_idx) {
        case move_north_idx:
            if (y < y_size - 1 && x < x_size - 1 && y <= move_north_bound) {
                next_rob_loc = rob_loc_pre + 1;
            }
            break;
        case look_north_idx:
            if (y < y_size - 1 && x < x_size - 1 && x > 0 && y < north_bound - 1) {
                next_rob_loc = rob_loc_pre + 1;
            }
            break;
        case move_south_idx:
            if (y > 0 && x < x_size - 1) {
                next_rob_loc = rob_loc_pre - 1;
            };
            break;
        case look_south_idx:
            if (y > 0 && x < x_size - 1 && x > 0 && y < north_bound) {
                next_rob_loc = rob_loc_pre - 1;
            };
            break;
        case move_west_idx:
            if (x > 0 && x < x_size - 1 && y < north_bound) {
                next_rob_loc = rob_loc_pre - y_size;
            }
            break;
        case look_west_idx:
            if (x > 0 && x < x_size - 1 && y < north_bound) {
                next_rob_loc = rob_loc_pre - y_size;
            }
            break;
        case move_east_idx:
            if (x < x_size - 1) {
                next_rob_loc = rob_loc_pre + y_size;
            }
            break;
        case look_east_idx:
            if (x < x_size - 1) {
                next_rob_loc = rob_loc_pre + y_size;
            }
            break;
        default:
            MyUtils::printErrorMsg("unknown action idx: " + std::to_string(action_idx));
    }
    if (next_rob_loc == -1) {
        return rob_loc_pre;
    }
    return next_rob_loc;
}


pair<int, int> kitchen_belief::getXY(int loc_idx) {
    int x = floor(loc_idx / y_size);
    int y = loc_idx % y_size;
    return make_pair(x, y);
};

int kitchen_belief::getLocIdx(int x, int y) {
    return x * y_size + y;
}

int kitchen_belief::getStateIdx(int rob_loc_idx, int obs_locs_idx) {
    int obs_locs_list_size = obs_locs_list.size();
    return rob_loc_idx * obs_locs_list_size + obs_locs_idx;
}

int kitchen_belief::getRobotLocIdx(int state_idx) {
    return state_idx / obs_locs_list.size();
}
int kitchen_belief::getObsLocsIdx(int state_idx) {
    return state_idx % obs_locs_list.size();
}

void kitchen_belief::printBeliefGroupInfo(int step) {
    if (MyUtils::debugLevel < MyUtils::LEVEL_TWO_MSG) {
        return;
    }
    if (step >= belief_group_size.size()) {
        MyUtils::printDebugMsg("do not have belief group information for step " +
                               std::to_string(step), MyUtils::LEVEL_ONE_MSG);
    } else {
        int group_num = belief_group_size[step].size();
        MyUtils::printDebugMsg("belief var number: " + std::to_string(group_num),
                               MyUtils::LEVEL_TWO_MSG);
        for (int g = 0; g < group_num; ++g) {
            MyUtils::printDebugMsg("group " + std::to_string(g)
                                   + ": " + std::to_string(belief_group_size[step][g]),
                                   MyUtils::LEVEL_TWO_MSG);
        }
        auto &b_to_g_map = belief_to_group_maps[step];

        for (auto b_to_g : b_to_g_map) {
            int idx = b_to_g.first;
            int r = getRobotLocIdx(idx);
            int i = getObsLocsIdx(idx);
            int g_idx = b_to_g.second;
            MyUtils::printDebugMsg("(" + std::to_string(r)
                                   + ", " + std::to_string(i) + "): "
                                   + std::to_string(g_idx),
                                   MyUtils::LEVEL_TWO_MSG);
        }
    }
}


bool kitchen_belief::in_collision(int rob_loc, const vector<int> & obs_locs) {
    for (int obs_loc : obs_locs) {
        if (rob_loc == obs_loc) {
            return true;
        }
    }
    return false;
}


void kitchen_belief::updateGroupSplitIdsForAction(int action_idx,
                                                  unordered_map<int, string> &group_split_ids, int step) {
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];

    //STEP - the most important function
    for (auto b_to_g_pre : b_to_g_map_pre) {
        int pre_idx = b_to_g_pre.first;
        int r = getRobotLocIdx(pre_idx);
        auto xy = getXY(r);
        int x = xy.first;

        int i = getObsLocsIdx(pre_idx);
        int g_idx_pre = b_to_g_pre.second;

        string group_idx_str = std::to_string(g_idx_pre);
        // for move actions
        if (action_idx == move_north_idx || action_idx == move_south_idx
            || action_idx == move_west_idx || action_idx == move_east_idx) {
            int next_idx;
            next_idx = getStateIdx(getNextLoc(action_idx, r), i);
            if (next_idx != pre_idx) {
                if (group_split_ids.find(next_idx) == group_split_ids.end()) {
                    group_split_ids[next_idx] = "";
                }
                group_split_ids[next_idx] += "(" + group_idx_str + ")"
                                             + "[" + std::to_string(action_idx) + "]";
            }
        } else if (action_idx == look_north_idx || action_idx == look_south_idx
                   || action_idx == look_west_idx || action_idx == look_east_idx) {
            // for look actions
            if (see_obs_num < num_obs) {
                int look_loc;
                look_loc = getNextLoc(action_idx, r);
                if (look_loc != r && in_collision(look_loc, obs_locs_list[i])) {
                    if (group_split_ids.find(pre_idx) == group_split_ids.end()) {
                        group_split_ids[pre_idx] = "";
                    }
                    group_split_ids[pre_idx] += "(" + group_idx_str + ")"
                                                + "[" + std::to_string(action_idx) + "]";
                }
            }
        } else if (action_idx == pick_up_left_hand_idx || action_idx == pick_up_right_hand_idx) {
            // for pickup actions
            if (x == x_size - 1) {
                int ready_idx = getStateIdx(ready_robot_loc, i);
                if (group_split_ids.find(ready_idx) == group_split_ids.end()) {
                    group_split_ids[ready_idx] = "";
                }
                group_split_ids[ready_idx] += "(" + group_idx_str + ")"
                                              + "[ready]";


                int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
                if (group_split_ids.find(unsafe_idx) == group_split_ids.end()) {
                    group_split_ids[unsafe_idx] = "";
                }
                group_split_ids[unsafe_idx] += "(" + group_idx_str + ")"
                                               + "[unsafe]";

                int goal_idx = getStateIdx(goal_robot_loc, i);
                if (group_split_ids.find(goal_idx) == group_split_ids.end()) {
                    group_split_ids[goal_idx] = "";
                }
                group_split_ids[goal_idx] += "(" + group_idx_str + ")"
                                             + "[goal]";
            }
        }
    }
}


void kitchen_belief::updateBeliefGroupMap(int step) {

    unordered_map<int, string> group_split_ids;

    updateGroupSplitIds(group_split_ids, step);

    // create new belief groups and mapping
    unordered_map<string, int> b_expr_to_g_map;
    belief_group_size.push_back(vector<int> ());
    belief_to_group_maps.push_back(unordered_map<int, int> ());
    int group_num = 0;

    for (auto g_split_id : group_split_ids) {
        int idx = g_split_id.first;
        string split_id = g_split_id.second;

        if (b_expr_to_g_map.find(group_split_ids[idx]) == b_expr_to_g_map.end()) {
            // this should be a new group
            b_expr_to_g_map[group_split_ids[idx]] = group_num;
            belief_group_size[step].push_back(0);
            ++ group_num;
        }
        int g_idx = b_expr_to_g_map[group_split_ids[idx]];
        ++ belief_group_size[step][g_idx];
        belief_to_group_maps[step][idx] = g_idx;
    }

    printBeliefGroupInfo(step);
}

void kitchen_belief::updateGroupSplitIds(unordered_map<int, string> &group_split_ids, int step) {
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];

    for (auto b_to_g_pre : b_to_g_map_pre) {
        int pre_idx = b_to_g_pre.first;
        int g_idx_pre = b_to_g_pre.second;
        string group_idx_str = std::to_string(g_idx_pre);
        if (group_split_ids.find(pre_idx) == group_split_ids.end()) {
            group_split_ids[pre_idx] = "";
        }
        group_split_ids[pre_idx] += ("(" + group_idx_str + ")");
    }

    if (MyUtils::solver_name == "partial") {
        if (a_start == look_north_idx) {
            if (o_start == positive_observation) {
                if (step == start_step + 1) {
                    updateGroupSplitIdsForAction(move_west_idx, group_split_ids, step);
                    return;
                } else if (step == start_step + 2) {
                    updateGroupSplitIdsForAction(move_north_idx, group_split_ids, step);
                    return;
                } else if (step == start_step + 3) {
                    updateGroupSplitIdsForAction(move_north_idx, group_split_ids, step);
                    return;
                }
            }
        }
    }
    updateGroupSplitIdsForAction(move_north_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_south_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_east_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_west_idx, group_split_ids, step);
    if (see_obs_num < num_obs) {
        if (MyUtils::solver_mode != "success") {
            updateGroupSplitIdsForAction(look_north_idx, group_split_ids, step);
        }
        updateGroupSplitIdsForAction(look_south_idx, group_split_ids, step);
        updateGroupSplitIdsForAction(look_east_idx, group_split_ids, step);
        updateGroupSplitIdsForAction(look_west_idx, group_split_ids, step);
    }
    updateGroupSplitIdsForAction(pick_up_left_hand_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(pick_up_right_hand_idx, group_split_ids, step);
}

vector<pair<int, double>> kitchen_belief::getObservationDistribution(
        int action_id, BeliefStatePtr b_pre, int step) {
    vector<pair<int, double>> observations;
    if (action_id == move_north_idx) {
        observations.push_back(make_pair (trans_observation, 1.0));
    } else if (action_id == move_south_idx) {
        observations.push_back(make_pair (trans_observation, 1.0));
    } else if (action_id == move_west_idx) {
        observations.push_back(make_pair (trans_observation, 1.0));
    } else if (action_id == move_east_idx) {
        observations.push_back(make_pair (trans_observation, 1.0));
    } else if (action_id == look_east_idx || action_id == look_north_idx
               || action_id == look_south_idx || action_id == look_west_idx) {
        double p_pos = 0.0;
        double p_neg = 0.0;
        auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
        vector<double> b_pre_vals = b_pre->getBeliefStateValues();
        for (int r = 0; r < rob_locs_size; ++r) {
            int look_loc = getNextLoc(action_id, r);
            for (int i = 0; i < obs_locs_list.size(); ++i) {
                int idx_pre = getStateIdx(r, i);
                if (b_to_g_map_pre.find(idx_pre) == b_to_g_map_pre.end()) {
                    continue;
                }
                int g_idx_pre = b_to_g_map_pre[idx_pre];
                double b_g_pre = b_pre_vals[g_idx_pre];

                if (look_loc == r) {
                    p_pos += b_g_pre * p_fp;
                    p_neg += b_g_pre * (1 - p_fp);
                } else {
                    if (in_collision(look_loc, obs_locs_list[i])) {
                        p_pos += b_g_pre * (1 - p_fn);
                        p_neg += b_g_pre * p_fn;
                    } else {
                        p_pos += b_g_pre * p_fp;
                        p_neg += b_g_pre * (1 - p_fp);
                    }
                }
            }
        }
        observations.push_back(make_pair(positive_observation, p_pos / (p_pos + p_neg)));
        observations.push_back(make_pair(negative_observation, p_neg / (p_pos + p_neg)));
    } else if (action_id == pick_up_left_hand_idx || action_id == pick_up_right_hand_idx) {
        double p_pos = 0.0;
        double p_neg = 0.0;
        double r_pos_prob;
        double r_neg_prob;
        double u_pos_prob;
        double u_neg_prob;
        double g_pos_prob;
        double g_neg_prob;
        if (action_id == pick_up_left_hand_idx) {
            r_pos_prob = 0.0;
            r_neg_prob = 0.0;
            u_pos_prob = 0.03;
            u_neg_prob = 0.07;
            g_pos_prob = 0.72;
            g_neg_prob = 0.18;
        } else if (action_id == pick_up_right_hand_idx) {
            r_pos_prob = 0.04;
            r_neg_prob = 0.01;
            u_pos_prob = 0.08;
            u_neg_prob = 0.02;
            g_pos_prob = 0.68;
            g_neg_prob = 0.17;
        }
        double other_pos_prob = r_pos_prob + u_pos_prob + g_pos_prob;
        double other_neg_prob = r_neg_prob + u_neg_prob + g_neg_prob;
        auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
        vector<double> b_pre_vals = b_pre->getBeliefStateValues();

        for (int r = 0; r < rob_locs_size; ++r) {
            auto xy = getXY(r);
            int x = xy.first;
            for (int i = 0; i < obs_locs_list.size(); ++i) {
                int idx_pre = getStateIdx(r, i);
                if (b_to_g_map_pre.find(idx_pre) == b_to_g_map_pre.end()) {
                    continue;
                }
                int g_idx= b_to_g_map_pre[idx_pre];
                double b_g_pre = b_pre_vals[g_idx];
                //FIXME - two conditions are same
                if (x == x_size - 1) {
                    // ready
                    p_pos += b_g_pre * r_pos_prob;
                    p_neg += b_g_pre * r_neg_prob;

                    // unsafe
                    p_pos += b_g_pre * u_pos_prob;
                    p_neg += b_g_pre * u_neg_prob;

                    // goal
                    p_pos += b_g_pre * g_pos_prob;
                    p_neg += b_g_pre * g_neg_prob;
                } else {
                    p_pos += b_g_pre * other_pos_prob;
                    p_neg += b_g_pre * other_neg_prob;
                }
            }
        }
        observations.push_back(make_pair(pickup_positive_observation, p_pos / (p_pos + p_neg)));
        observations.push_back(make_pair(pickup_negative_observation, p_neg / (p_pos + p_neg)));
    }
    return observations;
};

string kitchen_belief::getActionMeaning(int action_id) {
    string action_meaning;
    if (action_id == move_north_idx) {
        action_meaning = "move north";
    } else if (action_id == move_south_idx) {
        action_meaning = "move south";
    } else if (action_id == move_west_idx) {
        action_meaning = "move west";
    } else if (action_id == move_east_idx) {
        action_meaning = "move east";
    } else if (action_id == look_west_idx) {
        action_meaning = "look west";
    } else if (action_id == look_east_idx) {
        action_meaning = "look east";
    } else if (action_id == look_north_idx) {
        action_meaning = "look north";
    } else if (action_id == look_south_idx) {
        action_meaning = "look south";
    } else if (action_id == pick_up_left_hand_idx) {
        action_meaning = "pick up using left hand";
    } else if (action_id == pick_up_right_hand_idx) {
        action_meaning = "pick up using right hand";
    } else {
        MyUtils::printErrorMsg("unknown action idx: " + std::to_string(action_id));
    }
    return action_meaning;
}

string kitchen_belief::getObservationMeaning(int observation_id) {
    string observation_meaning;
    if (observation_id == trans_observation) {
        observation_meaning = "transition";
    } else if (observation_id == negative_observation) {
        observation_meaning = "no obstacle observed";
    } else if (observation_id == positive_observation) {
        observation_meaning = "obstacle observed";
    } else if (observation_id == pickup_negative_observation) {
        observation_meaning = "observe no object in hand";
    } else if (observation_id == pickup_positive_observation) {
        observation_meaning = "observe object in hand";
    } else {
        MyUtils::printErrorMsg("unknown observation id: "
                               + std::to_string(observation_id));
    }
    return observation_meaning;
}

pair<vector<double>, vector<double>> kitchen_belief::extractProbs(BeliefStatePtr b, int step) {
    double b_r_sum = 0.0;
    double b_obs_sum = 0.0;
    vector<double> b_r_vals;
    vector<double> b_obs_vals;
    for (int r = 0; r < rob_locs_size; ++r) {
        b_r_vals.push_back(0.0);
    }
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        b_obs_vals.push_back(0.0);
    }
    const vector<pair<double, double>> &b_vals
            = b->getBeliefStateExactValues();

    for (auto b_to_g : belief_to_group_maps[step]) {
        int idx = b_to_g.first;
        int r = getRobotLocIdx(idx);
        int i = getObsLocsIdx(idx);
        int g_idx = b_to_g.second;

        double n = b_vals[g_idx].first;
        double d = b_vals[g_idx].second;
        double b_r_i_k_val = n / d;
        b_r_vals[r] += b_r_i_k_val;
        b_r_sum += b_r_i_k_val;
        b_obs_vals[i] += b_r_i_k_val;
        b_obs_sum += b_r_i_k_val;
    }

    for (int r = 0; r < rob_locs_size; ++r) {
        int rob_loc = r;
        b_r_vals[rob_loc] = b_r_vals[rob_loc] / b_r_sum;
    }
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        b_obs_vals[i] = b_obs_vals[i] / b_obs_sum;
    }
    return make_pair(b_r_vals, b_obs_vals);
}


string kitchen_belief::beliefStateToString(BeliefStatePtr b, int step) {

    auto probs = extractProbs(b, step);
    auto &b_r_vals = probs.first;
    auto &b_obs_vals = probs.second;
    std::stringstream ss;
    ss << "robot probability: ";
    for (int r = 0; r < rob_locs_size; ++r) {
        int rob_loc = r;
        auto xy = getXY(rob_loc);
        int x = xy.first;
        int y = xy.second;
        double rob_prob = b_r_vals[rob_loc];
        ss << "[(" << x << ", " << y << "): " << rob_prob << "] ";
    }

    vector<double> obs_probs;

    for (int i = 0; i < obs_all_locs.size(); ++i) {
        obs_probs.push_back(0.0);
    }
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        double obs_i_prob = b_obs_vals[i];
        if (obs_i_prob == 0) {
            continue;
        }
        for (auto obs_loc : obs_locs_list[i]) {
            int idx = obs_loc - y_size;
            obs_probs[idx] += obs_i_prob;
//            SHARE->update_obs(idx,obs_probs[idx]);

        }
    }
    ss << "obstacle probability: ";
    for (int i = 0; i < obs_all_locs.size(); ++i) {
        auto xy = getXY(obs_all_locs[i]);
        ss << "[(" << xy.first << ", " << xy.second << "): ";
        ss << obs_probs[i] << "] ";

    }
    ss << endl;

    return ss.str();
}


int kitchen_belief::observe(BeliefStatePtr b, int action_id, int step) {
    if (action_id == move_north_idx) {
        return trans_observation;
    } else if (action_id == move_south_idx) {
        return trans_observation;
    } else if (action_id == move_west_idx) {
        return trans_observation;
    } else if (action_id == move_east_idx) {
        return trans_observation;
    } else if (action_id == look_east_idx || action_id == look_north_idx
               || action_id == look_south_idx || action_id == look_west_idx) {
        auto probs = extractProbs(b, step); // convert belief to obstacle probability
        auto &b_r_vals = probs.first;
        double max_rob_prob = 0.0;
        int curr_robot_loc = 0;
        for (int r = 0; r < rob_locs_size; ++r) {
            if (b_r_vals[r] > max_rob_prob) {
                max_rob_prob = b_r_vals[r];
                curr_robot_loc = r;
            }
        }
        bool see_obstacle = false;
        int look_loc = getNextLoc(action_id, curr_robot_loc);
        if (look_loc != curr_robot_loc) {
            for (auto obs_loc : obs_locs) {
                if (obs_loc == look_loc) {
                    see_obstacle = true;
                }
            }
        }
        double rand_point = MyUtils::uniform01();
        if (see_obstacle) {
            if (rand_point > p_fn) {
                MyUtils::printDebugMsg("Observe obstacle",  MyUtils::LEVEL_ONE_MSG);
                return positive_observation;
            } else {
                MyUtils::printDebugMsg("No obstacle observed (False Negative)",  MyUtils::LEVEL_ONE_MSG);
                return negative_observation;
            }
        } else {
            if (rand_point > p_fp) {
                MyUtils::printDebugMsg("No obstacle observed",  MyUtils::LEVEL_ONE_MSG);
                return negative_observation;
            } else {
                MyUtils::printDebugMsg("Observe obstacle (False Positive)",  MyUtils::LEVEL_ONE_MSG);
                return positive_observation;
            }
        }
    } else if (action_id == pick_up_left_hand_idx || action_id == pick_up_right_hand_idx) {
        int obs = MyUtils::uniformInt(0, 1);
        if (obs == 0) {
            return pickup_negative_observation;
        } else {
            return pickup_positive_observation;
        }
    }

    return 0;
}

void kitchen_belief::generateRandomInstances(string test_file_dir, int num) {
    // unordered_set<int> in_test;
    ofstream log_file;
    log_file.open(test_file_dir + "/corridor_test_" + to_string(1));
    log_file << x_size << " " << y_size << endl;
    log_file << delta << endl;
    log_file << num_obs << endl;
    for (auto obs_loc: obs_locs) {
        auto xy = getXY(obs_loc);
        log_file << xy.first <<  " " << xy.second << endl;
    }
    log_file.close();
    for (int i = 2; i <= num; ++i) {
        ofstream log_file;
        log_file.open(test_file_dir + "/corridor_test_" + to_string(i));
        log_file << x_size << " " << y_size << endl;
        log_file << delta << endl;
        log_file << num_obs << endl;
        int test_setup_idx = MyUtils::uniformInt(0, obs_locs_list.size() - 1);
        /*while (in_test.find(test_setup_idx) != in_test.end()) {
            test_setup_idx = MyUtils::uniformInt(0, obs_locs_list.size() - 1);
        }
        in_test.insert(test_setup_idx);*/
        MyUtils::printDebugMsg("Test " + to_string(i) + " obstacle locations index: " + to_string(test_setup_idx),  MyUtils::LEVEL_ONE_MSG);
        for (auto obs_loc: obs_locs_list[test_setup_idx]) {
            auto xy = getXY(obs_loc);
            log_file << xy.first <<  " " << xy.second << endl;
        }
        log_file.close();
    }
}

void kitchen_belief::computeAlphaVector(PolicyNodePtr policy) {
    unordered_map<int, double> alpha;
    int action = policy->getAction();

    auto child_nodes = policy->getChildNodes();

    if (MyUtils::debugLevel >= MyUtils::LEVEL_TWO_MSG) {
        if (!policy->isGoal()) {
            cout << "alpha vector for action " << getActionMeaning(action) << ":" << endl;
        } else {
            cout << "terminal alpha vector" << endl;
        }
    }

    // collision state -10
    for (int r = 0; r < rob_locs_size; ++r) {
        for (int i = 0; i < obs_locs_list.size(); ++i) {
            int idx = getStateIdx(r, i);
            double a_val = 0.0;

            for (int avoid_loc : avoid_regions) {
                if (r == avoid_loc) {
                    a_val = -10.0;
                }

            }

            /*if (in_collision(r, obs_locs_list[i])) {
                a_val = -100.0;
            }*/

            if (!policy->isGoal()) {
                // -1 for each action
                a_val += -1.0;
                if (action == look_east_idx || action == look_north_idx
                    || action == look_south_idx || action == look_west_idx) {
                    int look_loc = getNextLoc(action, r);
                    PolicyNodePtr pos_child = child_nodes[positive_observation];
                    PolicyNodePtr neg_child = child_nodes[negative_observation];

                    if (look_loc == r) {
                        a_val += p_fp * pos_child->getAlpha(idx);
                        a_val += (1 - p_fp) * neg_child->getAlpha(idx);
                    } else {
                        if (in_collision(look_loc, obs_locs_list[i])) {
                            a_val += (1 - p_fn) * pos_child->getAlpha(idx);
                            a_val += p_fn * neg_child->getAlpha(idx);;
                        } else {
                            a_val += p_fp * pos_child->getAlpha(idx);
                            a_val += (1 - p_fp) * neg_child->getAlpha(idx);
                        }
                    }
                } else if (action == move_north_idx || action == move_south_idx
                           || action == move_west_idx || action == move_east_idx) {
                    int next_loc = getNextLoc(action, r);
                    int idx_next = getStateIdx(next_loc, i);

                    PolicyNodePtr child = child_nodes[trans_observation];


                    a_val += p_succ * child->getAlpha(idx_next);
                    a_val += (1.0 - p_succ) * child->getAlpha(idx);
                } else if (action == pick_up_left_hand_idx || action == pick_up_right_hand_idx) {
                    double r_pos_prob;
                    double r_neg_prob;
                    double u_pos_prob;
                    double u_neg_prob;
                    double g_pos_prob;
                    double g_neg_prob;
                    if (action == pick_up_left_hand_idx) {
                        r_pos_prob = 0.0;
                        r_neg_prob = 0.0;
                        u_pos_prob = 0.03;
                        u_neg_prob = 0.07;
                        g_pos_prob = 0.72;
                        g_neg_prob = 0.18;
                    } else if (action == pick_up_right_hand_idx) {
                        r_pos_prob = 0.04;
                        r_neg_prob = 0.01;
                        u_pos_prob = 0.08;
                        u_neg_prob = 0.02;
                        g_pos_prob = 0.68;
                        g_neg_prob = 0.17;
                    }
                    double other_pos_prob = r_pos_prob + u_pos_prob + g_pos_prob;
                    double other_neg_prob = r_neg_prob + u_neg_prob + g_neg_prob;
                    PolicyNodePtr pos_child = child_nodes[pickup_positive_observation];
                    PolicyNodePtr neg_child = child_nodes[pickup_negative_observation];
                    int x = getXY(r).first;
                    if (x == x_size - 1) {
                        int ready_idx = getStateIdx(ready_robot_loc, i);

                        a_val += r_pos_prob * pos_child->getAlpha(ready_idx);
                        a_val += r_neg_prob * neg_child->getAlpha(ready_idx);

                        int unsafe_idx = getStateIdx(unsafe_robot_loc, i);

                        a_val += u_pos_prob * pos_child->getAlpha(unsafe_idx);
                        a_val += u_neg_prob * neg_child->getAlpha(unsafe_idx);

                        int goal_idx = getStateIdx(goal_robot_loc, i);

                        a_val += g_pos_prob * pos_child->getAlpha(goal_idx);
                        a_val += g_neg_prob * neg_child->getAlpha(goal_idx);

                    } else {
                        a_val += other_pos_prob * pos_child->getAlpha(idx);
                        a_val += other_neg_prob * neg_child->getAlpha(idx);
                    }
                }
            }
            if (abs(a_val) > MyUtils::DOUBLE_TOLENRANCE) {
                alpha[idx] = a_val;
                if (MyUtils::debugLevel >= MyUtils::LEVEL_TWO_MSG) {
                    cout << "for state: robot loc - " << r << ", obstacle locs - ( ";
                    for (auto obs_loc : obs_locs_list[i]){
                        cout << obs_loc << " ";
                    }
                    cout << "), alpha value: " << alpha[idx] << endl;
                }
            }
        }
    }
    policy->setAlpha(alpha);
}

double kitchen_belief::computeBeliefValue(BeliefStatePtr b, unordered_map<int, double> &alpha, int step) {
    auto &b_to_g_map = belief_to_group_maps[step];
    auto b_vals = b->getBeliefStateValues();
    double value = 0.0;

    for (auto alpha_i : alpha) {
        int idx = alpha_i.first;
        double alpha_i_v = alpha_i.second;
        if (b_to_g_map.find(idx) == b_to_g_map.end()) {
            continue;
        }
        int g_idx = b_to_g_map[idx];
        value += b_vals[g_idx] * alpha_i_v;

    }
    return value;
}

unordered_map<int, bool> kitchen_belief::getAvailableActions() {
    unordered_map<int, bool> actions;
    actions[move_north_idx] = true;
    actions[move_south_idx] = true;
    actions[move_west_idx] = true;
    actions[move_east_idx] = true;
    actions[look_north_idx] = true;
     actions[look_south_idx] = true;
     actions[look_west_idx] = true;
    actions[look_east_idx] = true;
    actions[pick_up_left_hand_idx] = true;
    actions[pick_up_right_hand_idx] = true;
    return actions;
};

BeliefStatePtr kitchen_belief::getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
                                                   int obs_id, int step) {
    // update group map
    if (step > belief_group_size.size()) {
        MyUtils::printErrorMsg("incorrect belief group map update");
    }
    if (step == belief_group_size.size()) {
        updateBeliefGroupMap(step);
    }

    if (isGoal(b_pre, step - 1)) {
        return nullptr;
    }


    auto b_vals = b_pre->getBeliefStateValues();
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    auto &b_to_g_map_next = belief_to_group_maps[step];

    double b_sum_pre = 0;
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        b_sum_pre += b_vals[g] * belief_group_size[step - 1][g];
    }

    if (action_id == move_north_idx || action_id == move_south_idx
        || action_id == move_west_idx || action_id == move_east_idx) {

        double r_top = 0;
        double r_bottom = 0;
        double r_left = 0;
        double r_right = 0;

        // TODO sanity check if the robot goes outside of the gird
        for (auto b_to_g : b_to_g_map_pre) {
            int idx = b_to_g.first;
            int r = getRobotLocIdx(idx);
            int g_idx_pre = b_to_g.second;

            double b_g_pre = b_vals[g_idx_pre];
            if (getNextLoc(move_west_idx, r) == r) {
                r_left += b_g_pre;
            }
            if (getNextLoc(move_east_idx, r) == r) {
                r_right += b_g_pre;
            }
            if (getNextLoc(move_south_idx, r) == r) {
                r_bottom += b_g_pre;
            }
            if (getNextLoc(move_north_idx, r) == r) {
                r_top += b_g_pre;
            }
        }

        if ((action_id == move_north_idx && r_top > reach_th * b_sum_pre)
            || (action_id == move_south_idx && r_bottom > reach_th * b_sum_pre)
            || (action_id == move_west_idx && r_left > reach_th * b_sum_pre)
            || (action_id == move_east_idx && r_right > reach_th * b_sum_pre)) {
            return nullptr;
        }

        // TODO add transition effect
        size_t B_G_N = belief_group_size[step].size(); // belief group number
        vector<double> move_effects(B_G_N, 0.0);
        vector<bool> has_pre(B_G_N, false);
        vector<bool> has_next(B_G_N, false);

        for (auto b_to_g : b_to_g_map_pre) {
            int idx_pre = b_to_g.first;
            int r = getRobotLocIdx(idx_pre);
            int r_next = getNextLoc(action_id, r);
            int i = getObsLocsIdx(idx_pre);
            int g_idx_pre = b_to_g.second;

            double b_g_pre = b_vals[g_idx_pre];
            if (r_next == r) {
                // no effect
                if (!has_pre[b_to_g_map_next[idx_pre]]) {
                    move_effects[b_to_g_map_next[idx_pre]] += b_g_pre;
                    has_pre[b_to_g_map_next[idx_pre]] = true;
                }
            } else {
                // success effect
                int idx_next = getStateIdx(r_next, i);
                if (!has_next[b_to_g_map_next[idx_next]]) {
                    move_effects[b_to_g_map_next[idx_next]] += (b_g_pre * p_succ);
                    has_next[b_to_g_map_next[idx_next]] = true;
                }
                // fail effect
                if (!has_pre[b_to_g_map_next[idx_pre]]) {
                    move_effects[b_to_g_map_next[idx_pre]] += (b_g_pre * (1 - p_succ));
                    has_pre[b_to_g_map_next[idx_pre]] = true;
                }
            }
        }

        vector<double> b_next_vals(B_G_N, 0.0);
        for (int g = 0; g < belief_group_size[step].size(); ++g) {
            b_next_vals[g] = move_effects[g];
        }
        // check safety
        double unsafe_prob = 0;
        double b_sum = 0;

        for (auto b_to_g : b_to_g_map_next) {
            int idx = b_to_g.first;
            int r = getRobotLocIdx(idx);
            int i = getObsLocsIdx(idx);
            int g_idx = b_to_g.second;

            if (in_collision(r, obs_locs_list[i])) {
                unsafe_prob += b_next_vals[g_idx];
            }
            b_sum += b_next_vals[g_idx];

        }

        if (unsafe_prob < delta * b_sum) {
            for (int g = 0; g < belief_group_size[step].size(); ++g) {
                b_next_vals[g] = b_next_vals[g] / b_sum;
            }
            return make_shared <BeliefState> (b_next_vals);
        }
    } else if (action_id == look_north_idx || action_id == look_south_idx
               || action_id == look_west_idx || action_id == look_east_idx) {
        double potential_unsafe_prob = 0;

        auto b_vals = b_pre->getBeliefStateValues();
        for (auto b_to_g : b_to_g_map_pre) {
            int idx = b_to_g.first;
            int r = getRobotLocIdx(idx);
            int look_loc =getNextLoc(action_id, r);
            if (look_loc == r) {
                // no effect for look
                continue;
            }
            int i = getObsLocsIdx(idx);
            int g_idx_pre = b_to_g.second;

            if (in_collision(look_loc, obs_locs_list[i])) {
                potential_unsafe_prob += b_vals[g_idx_pre];
            }

        }

        if (potential_unsafe_prob > delta * b_sum_pre && potential_unsafe_prob < 0.6 * b_sum_pre) {
            auto &b_to_g_map_next = belief_to_group_maps[step];
            vector<double> look_pos_effects;
            vector<double> look_neg_effects;
            for (int g = 0; g < belief_group_size[step].size(); ++g) {
                look_pos_effects.push_back(0);
                look_neg_effects.push_back(0);
            }

            for (auto b_to_g : b_to_g_map_pre) {
                int idx_pre = b_to_g.first;
                int r = getRobotLocIdx(idx_pre);
                int look_loc = getNextLoc(action_id, r);
                int i = getObsLocsIdx(idx_pre);
                int g_idx_pre = b_to_g.second;

                int g_idx_next = b_to_g_map_next[idx_pre];

                double b_g_pre = b_vals[g_idx_pre];
                if (look_loc == r) {
                    look_pos_effects[g_idx_next] = (b_g_pre * p_fp);
                    look_neg_effects[g_idx_next] = (b_g_pre * (1 - p_fp));
                } else {
                    if (in_collision(look_loc, obs_locs_list[i])) {
                        look_pos_effects[g_idx_next] = (b_g_pre * (1 - p_fn));
                        look_neg_effects[g_idx_next] = (b_g_pre * p_fn);
                    } else {
                        look_pos_effects[g_idx_next] = (b_g_pre * p_fp);
                        look_neg_effects[g_idx_next] = (b_g_pre * (1 - p_fp));

                    }
                }
            }

            vector<double> b_next_vals;
            double b_sum = 0;
            if (obs_id == positive_observation) {
                for (int g = 0; g < belief_group_size[step].size(); ++g) {
                    b_next_vals.push_back(look_pos_effects[g]);
                    b_sum += b_next_vals[g] * belief_group_size[step][g];
                }
            } else if (obs_id == negative_observation) {
                for (int g = 0; g < belief_group_size[step].size(); ++g) {
                    b_next_vals.push_back(look_neg_effects[g]);
                    b_sum += b_next_vals[g] * belief_group_size[step][g];
                }
            }
            for (int g = 0; g < belief_group_size[step].size(); ++g) {
                b_next_vals[g] = b_next_vals[g] / b_sum;
            }
            return make_shared <BeliefState> (b_next_vals);
        }
    } else if (action_id == pick_up_left_hand_idx || action_id == pick_up_right_hand_idx){
        double ready_prob = 0;
        for (int r = 0; r < rob_locs_size; ++r) {
            auto xy = getXY(r);
            int x = xy.first;
            if (x != x_size - 1) {
                continue;
            }
            for (int i = 0; i < obs_locs_list.size(); ++i) {
                int idx = getStateIdx(r, i);
                if (b_to_g_map_pre.find(idx) == b_to_g_map_pre.end()) {
                    continue;
                }
                int g_idx = b_to_g_map_pre[idx];
                ready_prob += b_vals[g_idx];
            }
        }
        if (ready_prob > reach_th * b_sum_pre) {
            double r_pos_prob;
            double r_neg_prob;
            double u_pos_prob;
            double u_neg_prob;
            double g_pos_prob;
            double g_neg_prob;
            if (action_id == pick_up_left_hand_idx) {
                r_pos_prob = 0.0;
                r_neg_prob = 0.0;
                u_pos_prob = 0.03;
                u_neg_prob = 0.07;
                g_pos_prob = 0.72;
                g_neg_prob = 0.18;
            } else if (action_id == pick_up_right_hand_idx) {
                r_pos_prob = 0.04;
                r_neg_prob = 0.01;
                u_pos_prob = 0.08;
                u_neg_prob = 0.02;
                g_pos_prob = 0.68;
                g_neg_prob = 0.17;
            }
            double other_pos_prob = r_pos_prob + u_pos_prob + g_pos_prob;
            double other_neg_prob = r_neg_prob + u_neg_prob + g_neg_prob;

            vector<bool> g_pre_used;
            for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
                g_pre_used.push_back(false);
            }
            vector<double> pickup_pos_effects;
            vector<double> pickup_neg_effects;

            for (int g = 0; g < belief_group_size[step].size(); ++g) {
                pickup_pos_effects.push_back(0);
                pickup_neg_effects.push_back(0);
            }

            for (auto b_to_g : b_to_g_map_pre) {
                int idx_pre = b_to_g.first;
                int r = getRobotLocIdx(idx_pre);
                int i = getObsLocsIdx(idx_pre);
                int g_idx_pre = b_to_g.second;

                auto xy = getXY(r);
                int x = xy.first;

                if (g_pre_used[g_idx_pre]) {
                    continue;
                }
                g_pre_used[g_idx_pre] = true;
                double b_g_pre = b_vals[g_idx_pre] * belief_group_size[step - 1][g_idx_pre];
                if (x == x_size - 1) {
                    int ready_idx = getStateIdx(ready_robot_loc, i);
                    int g_ready_idx = b_to_g_map_next[ready_idx];
                    pickup_pos_effects[g_ready_idx] += (b_g_pre * r_pos_prob);
                    pickup_neg_effects[g_ready_idx] += (b_g_pre * r_neg_prob);

                    int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
                    int g_unsafe_idx = b_to_g_map_next[unsafe_idx];
                    pickup_pos_effects[g_unsafe_idx] += (b_g_pre * u_pos_prob);
                    pickup_neg_effects[g_unsafe_idx] += (b_g_pre * u_neg_prob);

                    int goal_idx = getStateIdx(goal_robot_loc, i);
                    int g_goal_idx = b_to_g_map_next[goal_idx];
                    pickup_pos_effects[g_goal_idx] += (b_g_pre * g_pos_prob);
                    pickup_neg_effects[g_goal_idx] += (b_g_pre * g_neg_prob);
                } else {
                    int g_idx_next = b_to_g_map_next[idx_pre];
                    pickup_pos_effects[g_idx_next] += (b_g_pre * other_pos_prob);
                    pickup_neg_effects[g_idx_next] += (b_g_pre * other_neg_prob);
                }
            }

            vector<double> b_next_vals;
            double b_sum = 0;
            if (obs_id == pickup_positive_observation) {
                for (int g = 0; g < belief_group_size[step].size(); ++g) {
                    b_next_vals.push_back(pickup_pos_effects[g] / belief_group_size[step][g]);
                    b_sum += pickup_pos_effects[g];
                }
            } else {
                for (int g = 0; g < belief_group_size[step].size(); ++g) {
                    b_next_vals.push_back(pickup_neg_effects[g] / belief_group_size[step][g]);
                    b_sum += pickup_neg_effects[g];
                }
            }
            double goal_prob = 0;
            double unsafe_prob = 0;
            double ready_prob = 0;
            for (int i = 0; i < obs_locs_list.size(); ++i) {
                int goal_idx = getStateIdx(goal_robot_loc, i);
                if (b_to_g_map_next.find(goal_idx) != b_to_g_map_next.end()) {
                    int g_goal_idx = b_to_g_map_next[goal_idx];
                    goal_prob += b_next_vals[g_goal_idx];
                }
                int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
                if (b_to_g_map_next.find(unsafe_idx) != b_to_g_map_next.end()) {
                    int g_unsafe_idx = b_to_g_map_next[unsafe_idx];
                    unsafe_prob += b_next_vals[g_unsafe_idx];
                }
                int ready_idx = getStateIdx(ready_robot_loc, i);
                if (b_to_g_map_next.find(ready_idx) != b_to_g_map_next.end()) {
                    int g_ready_idx = b_to_g_map_next[ready_idx];
                    ready_prob += b_next_vals[g_ready_idx];
                }
            }
            if (unsafe_prob < 0.2 * (goal_prob + unsafe_prob + ready_prob)) {
                for (int g = 0; g < belief_group_size[step].size(); ++g) {
                    b_next_vals[g] = b_next_vals[g] / b_sum;
                }
                return make_shared <BeliefState> (b_next_vals);
            }
        }
    }
    return nullptr;
}

bool kitchen_belief::isGoal(BeliefStatePtr b, int step) {
    double goal_prob = 0;
    double unsafe_prob = 0;
    double ready_prob = 0;
    auto &b_to_g_map = belief_to_group_maps[step];
    vector<double> b_vals = b->getBeliefStateValues();
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        int goal_idx = getStateIdx(goal_robot_loc, i);
        if (b_to_g_map.find(goal_idx) != b_to_g_map.end()) {
            int g_goal_idx = b_to_g_map[goal_idx];
            goal_prob += b_vals[g_goal_idx];
        }
        int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
        if (b_to_g_map.find(unsafe_idx) != b_to_g_map.end()) {
            int g_unsafe_idx = b_to_g_map[unsafe_idx];
            unsafe_prob += b_vals[g_unsafe_idx];
        }
        int ready_idx = getStateIdx(ready_robot_loc, i);
        if (b_to_g_map.find(ready_idx) != b_to_g_map.end()) {
            int g_ready_idx = b_to_g_map[ready_idx];
            ready_prob += b_vals[g_ready_idx];
        }
    }
    if (goal_prob > reach_th * (goal_prob + unsafe_prob + ready_prob)) {
        return true;
    }
    return false;
}


void kitchen_belief::init(bool run_init) {

    // all possible obstacle locations
    for (int x = 1; x < x_size - 1; ++x) {
        for (int y = 0; y < y_size; ++y) {
            obs_all_locs.push_back(getLocIdx(x, y));
        }
    }
    if (run_init) {
        avoid_regions = {2, 3, 4, 5, 12, 15, 16, 25, 27, 28, 29};
        obs_all_locs.push_back(getLocIdx(x_size - 1, 0));
        obs_all_locs.push_back(getLocIdx(x_size - 1, 1));
    }

    vector<int> obs_locs_temp;
    genObstacleLocsList(0, obs_locs_temp, 0, obs_all_locs, obs_locs_list);
    MyUtils::printDebugMsg("states number: " + std::to_string(rob_locs_size * obs_locs_list.size()),
                           MyUtils::LEVEL_ONE_MSG);

    MyUtils::printDebugMsg("obstacle number: " + std::to_string(obs_locs_list.size()),
                           MyUtils::LEVEL_ONE_MSG);

    for (int k = 0; k <= start_step; ++k) {
        belief_group_size.push_back(vector<int> ());
        belief_to_group_maps.push_back(unordered_map<int, int> ());
    }
}
