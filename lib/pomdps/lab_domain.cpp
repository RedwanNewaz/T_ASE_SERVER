//
// Created by Yue Wang on 6/16/18.
//

#include "pomdps/lab_domain.h"
#include "my_utils.h"
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

LabDomain::LabDomain(string test_file_path):
        CorridorDomain(test_file_path, false) {
    init();
}

LabDomain::LabDomain(int s, BeliefStatePtr init_b,
                               int a_s, int o_s, unordered_map<int, int> init_b_to_g_map):
        CorridorDomain(s, init_b, a_s, o_s, init_b_to_g_map, false) {
    init();

}
void LabDomain::init() {
    avoid_regions = {getLocIdx(0, 2), getLocIdx(1, 0), getLocIdx(3, 1)};
    p_fp = 0.03;
    p_fn = 0.06;
    p_succ = 0.98;
}

POMDPDomainPtr LabDomain::createNewPOMDPDomain(int s, BeliefStatePtr new_init_b,
                                                    int a_s, int o_s) {
    return make_shared <LabDomain> (s, filterZeroProb(new_init_b, s),
                                    a_s, o_s, belief_to_group_maps[s]);
}

int LabDomain::getNextLoc(int action_idx, int rob_loc_pre) {
    if (rob_loc_pre == ready_robot_loc) {
        return rob_loc_pre;
    }
    auto xy = getXY(rob_loc_pre);
    int x = xy.first;
    int y = xy.second;
    int next_rob_loc = -1;
    switch (action_idx) {
        case move_north_idx:
            if (y < y_size - 1 && x < x_size - 1) {
                next_rob_loc = rob_loc_pre + 1;
            }
            break;
        case look_north_idx:
            if (y < y_size - 1 && x < x_size - 1 && x > 0) {
                next_rob_loc = rob_loc_pre + 1;
            }
            break;
        case move_south_idx:
            if (y > 0 && x < x_size - 1) {
                next_rob_loc = rob_loc_pre - 1;
            };
            break;
        case look_south_idx:
            if (y > 0 && x < x_size - 1 && x > 0) {
                next_rob_loc = rob_loc_pre - 1;
            };
            break;
        case move_west_idx:
            if (x > 0 && x < x_size - 1) {
                next_rob_loc = rob_loc_pre - y_size;
            }
            break;
        case look_west_idx:
            if (x > 0 && x < x_size - 1) {
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

void LabDomain::updateGroupSplitIds(unordered_map<int, string> &group_split_ids, int step) {
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    for (int r = 0; r < rob_locs_size; ++r) {
        auto xy = getXY(r);
        int x = xy.first;
        for (int i = 0; i < obs_locs_list.size(); ++i) {
            int pre_idx = getStateIdx(r, i);
            if (b_to_g_map_pre.find(pre_idx) == b_to_g_map_pre.end()) {
                break;
            }
            int g_idx_pre = b_to_g_map_pre[pre_idx];
            string group_idx_str = std::to_string(g_idx_pre);
            if (group_split_ids.find(pre_idx) == group_split_ids.end()) {
                group_split_ids[pre_idx] = "";
            }
            group_split_ids[pre_idx] += ("(" + group_idx_str + ")");
        }
    }

    updateGroupSplitIdsForAction(move_north_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_south_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_west_idx, group_split_ids, step);
    updateGroupSplitIdsForAction(move_east_idx, group_split_ids, step);
    {
        updateGroupSplitIdsForAction(look_north_idx, group_split_ids, step);
        updateGroupSplitIdsForAction(look_south_idx, group_split_ids, step);
        updateGroupSplitIdsForAction(look_west_idx, group_split_ids, step);
        updateGroupSplitIdsForAction(look_east_idx, group_split_ids, step);
    }
    updateGroupSplitIdsForAction(pick_up_idx, group_split_ids, step);
}

z3::expr LabDomain::genTransCond(z3::context &ctx, int step) {
    z3::expr_vector trans_conds(ctx);
    if (step > belief_group_size.size()) {
        MyUtils::printErrorMsg("incorrect belief group map update");
    }
    if (step == belief_group_size.size()) {
        updateBeliefGroupMap(step);
    }

    trans_conds.push_back(genMoveActionTransCond(ctx, move_north_idx,step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_south_idx, step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_west_idx, step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_east_idx,step));
    {
        trans_conds.push_back(genLookActionTransCond(ctx, look_north_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_south_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_west_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_east_idx, step));
    }
    trans_conds.push_back(genPickUpTransCond(ctx, pick_up_idx, step));
    return mk_or(trans_conds);
}

z3::expr LabDomain::genGoalCond(z3::context &ctx, int step) {
    z3::expr_vector goal_states(ctx);
    z3::expr_vector unsafe_states(ctx);
    z3::expr_vector ready_states(ctx);
    auto &b_to_g_map = belief_to_group_maps[step];
    vector<int> goal_g;
    vector<int> unsafe_g;
    vector<int> ready_g;
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        goal_g.push_back(0);
        unsafe_g.push_back(0);
        ready_g.push_back(0);
    }
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        int goal_idx = getStateIdx(goal_robot_loc, i);
        if (b_to_g_map.find(goal_idx) != b_to_g_map.end()) {
            int g_goal_idx = b_to_g_map[goal_idx];
            ++goal_g[g_goal_idx];
        }
        int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
        if (b_to_g_map.find(unsafe_idx) != b_to_g_map.end()) {
            int g_unsafe_idx = b_to_g_map[unsafe_idx];
            ++unsafe_g[g_unsafe_idx];
        }
        int ready_idx = getStateIdx(ready_robot_loc, i);
        if (b_to_g_map.find(ready_idx) != b_to_g_map.end()) {
            int g_ready_idx = b_to_g_map[ready_idx];
            ++ready_g[g_ready_idx];
        }
    }
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        if (goal_g[g] != 0) {
            goal_states.push_back(b_g_k * goal_g[g]);
        }
        if (unsafe_g[g] != 0) {
            unsafe_states.push_back(b_g_k * unsafe_g[g]);
        }
        if (ready_g[g] != 0) {
            ready_states.push_back(b_g_k * ready_g[g]);
        }
    }
    if (goal_states.size() == 0) {
        return ctx.bool_val(false);
    }
    z3::expr_vector goal_conds(ctx);
    goal_conds.push_back(sum(goal_states) > ctx.real_val(to_string(reach_th).c_str())
                                            * (sum(goal_states) + sum(unsafe_states) + sum(ready_states)));

    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    goal_conds.push_back((a_k == ctx.int_val(pick_up_idx)));
    return mk_and(goal_conds);
}

string LabDomain::getActionMeaning(int action_id) {
    string action_meaning;
    if (action_id == pick_up_idx) {
        action_meaning = "pick up";
    } else {
        return CorridorDomain::getActionMeaning(action_id);
    }
    return action_meaning;
}


vector<double> LabDomain::getObsProbs(BeliefStatePtr b, int step) {
    auto probs = extractProbs(b, step);
    auto &b_obs_vals = probs.second;

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
        }
    }

    return obs_probs;
}

void LabDomain::generateRandomInstances(string test_file_dir, int num) {
    // unordered_set<int> in_test;
    ofstream log_file;
    log_file.open(test_file_dir + "/lab_test_" + to_string(1));
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
        log_file.open(test_file_dir + "/lab_test_" + to_string(i));
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

