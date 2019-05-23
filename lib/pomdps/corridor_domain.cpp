#ifndef CORRIDOR_DOMAIN_CC
#define CORRIDOR_DOMAIN_CC

#include "pomdps/corridor_domain.h"
#include "my_utils.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_set>


using namespace std;

int CorridorDomain::x_size = 0;
int CorridorDomain::y_size = 0;
int CorridorDomain::rob_locs_size = 0;
int CorridorDomain::ready_robot_loc = 0;
int CorridorDomain::unsafe_robot_loc = 0;
int CorridorDomain::goal_robot_loc = 0;
int CorridorDomain::num_obs = 0;
double CorridorDomain::delta = 0.0;
vector<int> CorridorDomain::obs_locs;

double CorridorDomain::p_fp = 0.02; // observation
double CorridorDomain::p_fn = 0.05; // observation
double CorridorDomain::p_succ = 0.98; //# transition probability

CorridorDomain::CorridorDomain(string test_file_path, bool run_init):
        POMDPDomain(0, 0, 0), obs_all_locs(), obs_locs_list(),
        belief_group_size(), belief_to_group_maps() {
    // for corridor domain, the paramter is the number
    // of cylinder obstacles in the environment

    // for each cylinder, we need to consider the probability distribution of
    // cylinder locations, constraints:
    // 1) only one cylinder per cell
    // 2) no cylinder in the first column

//    share_ = SHARE->getPtr();
    // read test description
    ifstream test_file;
    test_file.open(test_file_path);
    if (test_file.is_open())
    {
        test_file >> x_size >> y_size;
        rob_locs_size = x_size * y_size + 3; // plus states for pick-up actions
        ready_robot_loc = x_size * y_size;
        unsafe_robot_loc = ready_robot_loc + 1;//
        goal_robot_loc = unsafe_robot_loc + 1;
        test_file >> delta;
        test_file >> num_obs;
        for (int i = 0; i < num_obs; ++i) {
            int x, y;
            test_file >> x >> y;
            obs_locs.push_back(getLocIdx(x, y));
        }
        test_file >> epsilon;
        test_file.close();
    }
    else {
        MyUtils::printErrorMsg("Unable to open test file: " + test_file_path);
    }

    MyUtils::printDebugMsg("column number: " + to_string(x_size),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("row number: " + to_string(y_size),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("safety threshold: " + to_string(delta),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("obstacle number: " + to_string(num_obs),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("obstacle locations: ",  MyUtils::LEVEL_ONE_MSG);
    for (auto obs_loc : obs_locs) {
        auto xy = getXY(obs_loc);
        MyUtils::printDebugMsg("(" + to_string(xy.first) + ", " + to_string(xy.second) + ")",
                               MyUtils::LEVEL_ONE_MSG);
    }

    see_obs_num = 0;

    init(run_init);

    vector<pair<double, double>> init_b_vals;
    init_b_vals.push_back(make_pair(1.0, obs_locs_list.size()));
    init_belief = make_shared <BeliefState> (init_b_vals);
    
    belief_group_size[start_step].push_back(obs_locs_list.size());
    // only one belief group for rob_loc == init_loc
    // uniform distribution over all possible obstacle loc conbinations
    int g_idx = 0;
    // establish mapping
    init_robot_loc = 0;
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        int idx = getStateIdx(init_robot_loc, i);
        belief_to_group_maps[start_step][idx] = g_idx;
    }

}

CorridorDomain::CorridorDomain(int s, BeliefStatePtr init_b,
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


void CorridorDomain::init(bool run_init) {

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

BeliefStatePtr CorridorDomain::filterZeroProb(BeliefStatePtr b, int step) {

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

POMDPDomainPtr CorridorDomain::createNewPOMDPDomain(int s, BeliefStatePtr new_init_b,
                                                    int a_s, int o_s) {


    return make_shared <CorridorDomain> (s, filterZeroProb(new_init_b, s),
                                         a_s, o_s, belief_to_group_maps[s]);
}

void CorridorDomain::genObstacleLocsList(int obs_idx, vector<int> &obs_locs_temp, int loc_idx,
                                        const vector<int> &obs_all_locs,
                                        vector<vector<int>> &obs_locs_list) {
    /*
     obs = np.arange(12, 26)
     comb = combinations(obs, num_obs)
     return list(comb)
     */
    if (obs_idx == num_obs) {
        obs_locs_list.push_back(std::vector<int> (obs_locs_temp));
    } else {
        for (int i = loc_idx; i < obs_all_locs.size(); ++i) {
            obs_locs_temp.push_back(obs_all_locs[i]);
            genObstacleLocsList(obs_idx + 1, obs_locs_temp, i + 1, obs_all_locs, obs_locs_list);
            obs_locs_temp.pop_back();
        }
//        cout<< "obs_loc_list size " <<obs_locs_list.size()<<endl;
    }
}
    

int CorridorDomain::getNextLoc(int action_idx, int rob_loc_pre) {
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


pair<int, int> CorridorDomain::getXY(int loc_idx) {
    int x = floor(loc_idx / y_size);
    int y = loc_idx % y_size;
    return make_pair(x, y);
};

int CorridorDomain::getLocIdx(int x, int y) {
    return x * y_size + y;
}

int CorridorDomain::getStateIdx(int rob_loc_idx, int obs_locs_idx) {
    int obs_locs_list_size = obs_locs_list.size();
    return rob_loc_idx * obs_locs_list_size + obs_locs_idx;
}

int CorridorDomain::getRobotLocIdx(int state_idx) {
    return state_idx / obs_locs_list.size();
}
int CorridorDomain::getObsLocsIdx(int state_idx) {
    return state_idx % obs_locs_list.size();
}

void CorridorDomain::printBeliefGroupInfo(int step) {
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


z3::expr CorridorDomain::genInitCond(z3::context &ctx) {
    // assume uniform distribution for each obstacle
    // in a colunmn
    z3::expr_vector init_conds(ctx);

    vector<z3::expr> b_s_vars = getBeliefStateVarList(ctx, start_step);
    vector<z3::expr> b_s_vals = init_belief->getBeliefStateExprs(ctx);
    if (b_s_vars.size() != b_s_vals.size()) {
        MyUtils::printErrorMsg(
                "when generating initial condition, the sizes of belief vars and belief values do not agree"
        );
    }
    for (int i = 0; i < b_s_vals.size(); ++i) {
        init_conds.push_back(b_s_vars[i] == b_s_vals[i]);
    }
    string obs_num_var_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + "obs_num_"
                                 + std::to_string(start_step);
    z3::expr obs_num_var = ctx.int_const(obs_num_var_name.c_str());
    init_conds.push_back(obs_num_var == ctx.int_val(see_obs_num));

    string a_s_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(start_step);
    z3::expr a_s_var = ctx.int_const(a_s_name.c_str());
    init_conds.push_back(a_s_var == ctx.int_val(a_start));

    printBeliefGroupInfo(start_step);

    return mk_and(init_conds);
}


z3::expr CorridorDomain::genBeliefSum(z3::context &ctx, int step) {
    z3::expr_vector all_states(ctx);
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        all_states.push_back(b_g_k * ctx.real_val(belief_group_size[step][g]));
    }
    return sum(all_states);
}

bool CorridorDomain::in_collision(int rob_loc, const vector<int> & obs_locs) {
    for (int obs_loc : obs_locs) {
        if (rob_loc == obs_loc) {
            return true;
        }
    }
    return false;
}

z3::expr CorridorDomain::genMoveActionPreCond(z3::context &ctx, int action_idx, int step) {
    string rob_pos_pre_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 1);
    z3::expr rob_pos_pre = ctx.int_const(rob_pos_pre_name.c_str());
    z3::expr_vector pre_conds(ctx);

    // not goal
    pre_conds.push_back(!genGoalCond(ctx, step - 1));

    vector<int> r_left_g, r_right_g, r_top_g, r_bottom_g;
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        r_left_g.push_back(0);
        r_right_g.push_back(0);
        r_top_g.push_back(0);
        r_bottom_g.push_back(0);
    }
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];

    for (auto b_to_g : b_to_g_map_pre) {
        int idx_pre = b_to_g.first;
        int r = getRobotLocIdx(idx_pre);
        int g_idx_pre = b_to_g.second;

        if (getNextLoc(move_west_idx, r) == r) {
            ++r_left_g[g_idx_pre];
        }
        if (getNextLoc(move_east_idx, r) == r) {
            ++r_right_g[g_idx_pre];
        }
        if (getNextLoc(move_south_idx, r) == r) {
            ++r_bottom_g[g_idx_pre];
        }
        if (getNextLoc(move_north_idx, r) == r) {
            ++r_top_g[g_idx_pre];
        }
    }

    z3::expr_vector r_left(ctx);
    z3::expr_vector r_right(ctx);
    z3::expr_vector r_top(ctx);
    z3::expr_vector r_bottom(ctx);
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(g) + "_"
                                   + std::to_string(step - 1);
        z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
        if (r_left_g[g] != 0) {
            r_left.push_back(b_g_pre * r_left_g[g]);
        }
        if (r_right_g[g] != 0) {
            r_right.push_back(b_g_pre * r_right_g[g]);
        }
        if (r_bottom_g[g] != 0) {
            r_bottom.push_back(b_g_pre * r_bottom_g[g]);
        }
        if (r_top_g[g] != 0) {
            r_top.push_back(b_g_pre * r_top_g[g]);
        }
    }
    z3::expr b_sum_pre = genBeliefSum(ctx, step - 1);
    switch (action_idx) {
        case move_north_idx:
            if (r_top.size() != 0) {
                pre_conds.push_back(sum(r_top) <
                                    ctx.real_val(to_string(reach_th).c_str()) * b_sum_pre);
            }
            break;
        case move_south_idx:
            if (r_bottom.size() != 0) {
                pre_conds.push_back(sum(r_bottom) <
                                    ctx.real_val(to_string(reach_th).c_str()) * b_sum_pre);
            }
            break;
        case move_west_idx:
            if (r_left.size() != 0) {
                pre_conds.push_back(sum(r_left) <
                                    ctx.real_val(to_string(reach_th).c_str()) * b_sum_pre);
            }
            break;
        case move_east_idx:
            if (r_right.size() != 0) {
                pre_conds.push_back(sum(r_right) <
                                    ctx.real_val(to_string(reach_th).c_str()) * b_sum_pre);
            }
            break;
        default:
            MyUtils::printErrorMsg("unknown action idx: " + std::to_string(action_idx));
    }


    if (MyUtils::solver_name == "partial") {
        if (step > 2 && step < 4) {
            if (action_idx == move_north_idx) {
                string a_pre_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step - 1);
                z3::expr a_pre = ctx.int_const(a_pre_name.c_str());
                string a_pre_pre_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step - 2);
                z3::expr a_pre_pre = ctx.int_const(a_pre_pre_name.c_str());
                pre_conds.push_back(!(a_pre == ctx.int_val(move_north_idx)) ||
                                    (a_pre_pre == ctx.int_val(move_west_idx))
                );
            }
        }
    }


    return mk_and(pre_conds);
}

z3::expr CorridorDomain::genMoveActionTransCond(z3::context &ctx, int action_idx, int step) {
    z3::expr_vector move_trans(ctx);
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    move_trans.push_back(a_k == ctx.int_val(action_idx));
    move_trans.push_back(genMoveActionPreCond(ctx, action_idx, step));

    string o_k_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr o_k = ctx.int_const(o_k_name.c_str());
    move_trans.push_back(o_k == ctx.int_val(trans_observation));

    vector<z3::expr_vector> move_effects;
    vector<bool> has_pre;
    vector<bool> has_next;
    // initialize
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        move_effects.push_back(z3::expr_vector(ctx));
        has_pre.push_back(false);
        has_next.push_back(false);
    }

    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    auto &b_to_g_map_next = belief_to_group_maps[step];
    z3::expr p_succ_expr = ctx.real_val(std::to_string(p_succ).c_str());
    z3::expr p_fail_expr = ctx.real_val(std::to_string(1 - p_succ).c_str());

    for (auto b_to_g : b_to_g_map_pre) {
        int idx_pre = b_to_g.first;
        int r = getRobotLocIdx(idx_pre);
        int i = getObsLocsIdx(idx_pre);
        int r_next = getNextLoc(action_idx, r);
        int g_idx_pre = b_to_g.second;

        std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(g_idx_pre) + "_"
                                   + std::to_string(step - 1);
        z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
        if (r_next == r) {
            // no effect
            if (!has_pre[b_to_g_map_next[idx_pre]]) {
                move_effects[b_to_g_map_next[idx_pre]].push_back(b_g_pre);
                has_pre[b_to_g_map_next[idx_pre]] = true;
            }
        } else {
            // success effect
            int idx_next = getStateIdx(r_next, i);
            if (!has_next[b_to_g_map_next[idx_next]]) {
                move_effects[b_to_g_map_next[idx_next]].push_back(b_g_pre * p_succ_expr);
                has_next[b_to_g_map_next[idx_next]] = true;
            }
            // fail effect
            if (!has_pre[b_to_g_map_next[idx_pre]]) {
                move_effects[b_to_g_map_next[idx_pre]].push_back(b_g_pre * p_fail_expr);
                has_pre[b_to_g_map_next[idx_pre]] = true;
            }

        }
    }

    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        if (move_effects[g].size() == 0) {
            move_trans.push_back(b_g_k == ctx.real_val(0));
        } else {
            move_trans.push_back(b_g_k == sum(move_effects[g]));
        }

    }
    string obs_num_var_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                  + "obs_num_"
                                  + std::to_string(step - 1);
    z3::expr obs_num_var_pre = ctx.int_const(obs_num_var_pre_name.c_str());
    string obs_num_var_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                + "obs_num_"
                                + std::to_string(step);
    z3::expr obs_num_var_k = ctx.int_const(obs_num_var_k_name.c_str());
    move_trans.push_back(obs_num_var_k == obs_num_var_pre);

    move_trans.push_back(genMoveSafeCond(ctx, step));
    return mk_and(move_trans);
}

z3::expr CorridorDomain::genLookActionPreCond(z3::context &ctx, int action_idx, int step) {
    z3::expr_vector potential_unsafe_states(ctx);
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    vector<int> potential_unsafe_g;
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        potential_unsafe_g.push_back(0);
    }

    for (auto b_to_g : b_to_g_map_pre) {
        int idx_pre = b_to_g.first;
        int r = getRobotLocIdx(idx_pre);
        int look_loc =getNextLoc(action_idx, r);
        if (look_loc == r) {
            // no effect for look
            continue;
        }
        int i = getObsLocsIdx(idx_pre);
        int g_idx_pre = b_to_g.second;

        if (in_collision(look_loc, obs_locs_list[i])) {
            ++potential_unsafe_g[g_idx_pre];
        }
    }

    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        if (potential_unsafe_g[g] == 0) {
            continue;
        }
        std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(g) + "_"
                                   + std::to_string(step - 1);
        z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
        potential_unsafe_states.push_back(b_g_pre * potential_unsafe_g[g]);
    }

    if (potential_unsafe_states.size() == 0) {
        return ctx.bool_val(false);
    }
    z3::expr_vector pre_conds(ctx);

    pre_conds.push_back(!genGoalCond(ctx, step - 1));

    z3::expr collision_threshold = ctx.real_val(to_string(delta).c_str());
    z3::expr b_sum_pre = genBeliefSum(ctx, step - 1);
    pre_conds.push_back(sum(potential_unsafe_states) > collision_threshold * b_sum_pre);
    pre_conds.push_back(sum(potential_unsafe_states) < ctx.real_val(6, 10) * b_sum_pre);

    return mk_and(pre_conds);
}

z3::expr CorridorDomain::genLookActionTransCond(z3::context &ctx, int action_idx, int step) {
    z3::expr_vector look_trans(ctx);
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    look_trans.push_back(a_k == ctx.int_val(action_idx));

    look_trans.push_back(genLookActionPreCond(ctx, action_idx, step));

    string rob_pos_pre_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 1);
    z3::expr rob_pos_pre = ctx.int_const(rob_pos_pre_name.c_str());
    string rob_pos_k_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step);
    z3::expr rob_pos_k = ctx.int_const(rob_pos_k_name.c_str());

    z3::expr_vector look_effects(ctx);
    z3::expr p_fp_expr = ctx.real_val(to_string(p_fp).c_str());
    z3::expr p_fn_expr = ctx.real_val(to_string(p_fn).c_str());
    z3::expr p_1_fp_expr = ctx.real_val(to_string(1 - p_fp).c_str());
    z3::expr p_1_fn_expr = ctx.real_val(to_string((1 - p_fn)).c_str());
    string o_k_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr o_k = ctx.int_const(o_k_name.c_str());

    // positive observation for obstacle
    vector<z3::expr_vector> look_pos_effects;
    // negative observation - no obstacle observed
    vector<z3::expr_vector> look_neg_effects;

    // initialize
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        look_pos_effects.push_back(z3::expr_vector(ctx));
        look_neg_effects.push_back(z3::expr_vector(ctx));
    }
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    auto &b_to_g_map_next = belief_to_group_maps[step];

    for (auto b_to_g : b_to_g_map_pre) {
        int idx_pre = b_to_g.first;
        int r = getRobotLocIdx(idx_pre);
        int look_loc = getNextLoc(action_idx, r);
        int i = getObsLocsIdx(idx_pre);
        int g_idx_pre = b_to_g.second;
        int g_idx_next = b_to_g_map_next[idx_pre];
        if (look_pos_effects[g_idx_next].size() == 0) {
            std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                       + std::to_string(g_idx_pre) + "_"
                                       + std::to_string(step - 1);
            z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
            if (look_loc == r) {
                look_pos_effects[g_idx_next].push_back(b_g_pre * p_fp_expr);
            } else {
                if (in_collision(look_loc, obs_locs_list[i])) {
                    look_pos_effects[g_idx_next].push_back(b_g_pre * p_1_fn_expr);
                } else {
                    look_pos_effects[g_idx_next].push_back(b_g_pre * p_fp_expr);
                }
            }
        }
        if (look_neg_effects[g_idx_next].size() == 0) {
            std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                       + std::to_string(g_idx_pre) + "_"
                                       + std::to_string(step - 1);
            z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
            if (look_loc == r) {
                look_neg_effects[g_idx_next].push_back(b_g_pre * p_1_fp_expr);
            } else {
                if (in_collision(look_loc, obs_locs_list[i])) {
                    look_neg_effects[g_idx_next].push_back(b_g_pre * p_fn_expr);
                } else {
                    look_neg_effects[g_idx_next].push_back(b_g_pre * p_1_fp_expr);
                }
            }
        }
    }

    z3::expr_vector pos_conds(ctx);
    z3::expr_vector neg_conds(ctx);
    pos_conds.push_back(o_k == ctx.int_val(positive_observation));
    neg_conds.push_back(o_k == ctx.int_val(negative_observation));
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        if (look_pos_effects[g].size() == 0) {
            pos_conds.push_back(b_g_k == ctx.real_val(0));
        } else {
            pos_conds.push_back(b_g_k == (sum(look_pos_effects[g])));
        }
        if (look_neg_effects[g].size() == 0) {
            neg_conds.push_back(b_g_k == ctx.real_val(0));
        } else {
            neg_conds.push_back(b_g_k == (sum(look_neg_effects[g])));
        }
    }

    string obs_num_var_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                  + "obs_num_"
                                  + std::to_string(step - 1);
    z3::expr obs_num_var_pre = ctx.int_const(obs_num_var_pre_name.c_str());
    string obs_num_var_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                + "obs_num_"
                                + std::to_string(step);
    z3::expr obs_num_var_k = ctx.int_const(obs_num_var_k_name.c_str());
    pos_conds.push_back(obs_num_var_k == obs_num_var_pre + ctx.int_val(1));
    neg_conds.push_back(obs_num_var_k == obs_num_var_pre);


    look_effects.push_back(mk_and(pos_conds));
    look_effects.push_back(mk_and(neg_conds));
    look_trans.push_back(mk_or(look_effects));

    return mk_and(look_trans);
}

z3::expr CorridorDomain::genPickUpTransCond(z3::context &ctx, int action_idx, int step) {
    z3::expr_vector pick_up_trans(ctx);
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    pick_up_trans.push_back(a_k == ctx.int_val(action_idx));

    z3::expr_vector ready_states(ctx);
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    vector<int> ready_g;
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        ready_g.push_back(0);
    }
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
            ++ready_g[g_idx];
        }
    }

    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        if (ready_g[g] == 0) {
            continue;
        }
        std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step - 1);
        z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str());
        ready_states.push_back(b_g_pre * ready_g[g]);
    }
    if (ready_states.size() == 0) {
        return ctx.bool_val(false);
    }
    pick_up_trans.push_back(sum(ready_states) >
                                    ctx.real_val(to_string(reach_th).c_str())
                                    * genBeliefSum(ctx, step - 1));

    // effect
    z3::expr_vector pickup_effects(ctx);
    // positive observation for pickup
    vector<z3::expr_vector> pickup_pos_effects;
    // negative observation for pickup
    vector<z3::expr_vector> pickup_neg_effects;
    // initialize
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        pickup_pos_effects.push_back(z3::expr_vector(ctx));
        pickup_neg_effects.push_back(z3::expr_vector(ctx));
    }
    z3::expr r_pos_prob(ctx);
    z3::expr r_neg_prob(ctx);
    z3::expr u_pos_prob(ctx);
    z3::expr u_neg_prob(ctx);
    z3::expr g_pos_prob(ctx);
    z3::expr g_neg_prob(ctx);
    z3::expr other_pos_prob(ctx);
    z3::expr other_neg_prob(ctx);
    if (action_idx == pick_up_left_hand_idx) {
        r_pos_prob = ctx.real_val(0);
        r_neg_prob = ctx.real_val(0);
        u_pos_prob = ctx.real_val(3, 100);
        u_neg_prob = ctx.real_val(7, 100);
        g_pos_prob = ctx.real_val(72, 100);
        g_neg_prob = ctx.real_val(18, 100);
        other_pos_prob = ctx.real_val(3 + 72, 100);
        other_neg_prob = ctx.real_val(7 + 18, 100);
    } else if (action_idx == pick_up_right_hand_idx) {
        r_pos_prob = ctx.real_val(4, 100);
        r_neg_prob = ctx.real_val(1, 100);
        u_pos_prob = ctx.real_val(8, 100);
        u_neg_prob = ctx.real_val(2, 100);
        g_pos_prob = ctx.real_val(68, 100);
        g_neg_prob = ctx.real_val(17, 100);
        other_pos_prob = ctx.real_val(4 + 8 + 68, 100);
        other_neg_prob = ctx.real_val(1 + 2 + 17, 100);
    }
    auto &b_to_g_map_next = belief_to_group_maps[step];
    vector<bool> g_pre_used;
    for (int g = 0; g < belief_group_size[step - 1].size(); ++g) {
        g_pre_used.push_back(false);
    }
    for (int r = 0; r < rob_locs_size; ++r) {
        auto xy = getXY(r);
        int x = xy.first;
        for (int i = 0; i < obs_locs_list.size(); ++i) {
            int idx_pre = getStateIdx(r, i);
            if (b_to_g_map_pre.find(idx_pre) == b_to_g_map_pre.end()) {
                continue;
            }
            int g_idx_pre = b_to_g_map_pre[idx_pre];
            if (g_pre_used[g_idx_pre]) {
                continue;
            }
            g_pre_used[g_idx_pre] = true;
            std::string b_g_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                       + std::to_string(g_idx_pre) + "_"
                                       + std::to_string(step - 1);
            z3::expr b_g_pre = ctx.real_const(b_g_pre_name.c_str())
                               * ctx.real_val(belief_group_size[step - 1][g_idx_pre]);
            if (x == x_size - 1) {
                int ready_idx = getStateIdx(ready_robot_loc, i);
                int g_ready_idx = b_to_g_map_next[ready_idx];
                pickup_pos_effects[g_ready_idx].push_back(b_g_pre * r_pos_prob);
                pickup_neg_effects[g_ready_idx].push_back(b_g_pre * r_neg_prob);

                int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
                int g_unsafe_idx = b_to_g_map_next[unsafe_idx];
                pickup_pos_effects[g_unsafe_idx].push_back(b_g_pre * u_pos_prob);
                pickup_neg_effects[g_unsafe_idx].push_back(b_g_pre * u_neg_prob);

                int goal_idx = getStateIdx(goal_robot_loc, i);
                int g_goal_idx = b_to_g_map_next[goal_idx];
                pickup_pos_effects[g_goal_idx].push_back(b_g_pre * g_pos_prob);
                pickup_neg_effects[g_goal_idx].push_back(b_g_pre * g_neg_prob);
            } else {
                int g_idx_next = b_to_g_map_next[idx_pre];
                pickup_pos_effects[g_idx_next].push_back(b_g_pre * other_pos_prob);
                pickup_neg_effects[g_idx_next].push_back(b_g_pre * other_neg_prob);
            }
        }
    }

    z3::expr_vector pos_conds(ctx);
    z3::expr_vector neg_conds(ctx);
    string o_k_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr o_k = ctx.int_const(o_k_name.c_str());
    pos_conds.push_back(o_k == ctx.int_val(pickup_positive_observation));
    neg_conds.push_back(o_k == ctx.int_val(pickup_negative_observation));
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        if (pickup_pos_effects[g].size() == 0) {
            pos_conds.push_back(b_g_k == ctx.real_val(0));
        } else {
            pos_conds.push_back(b_g_k * ctx.real_val(belief_group_size[step][g])
                                == (sum(pickup_pos_effects[g])));
        }
        if (pickup_neg_effects[g].size() == 0) {
            neg_conds.push_back(b_g_k == ctx.real_val(0));
        } else {
            neg_conds.push_back(b_g_k * ctx.real_val(belief_group_size[step][g])
                                == (sum(pickup_neg_effects[g])));
        }
    }
    pickup_effects.push_back(mk_and(pos_conds));
    pickup_effects.push_back(mk_and(neg_conds));

    pick_up_trans.push_back(mk_or(pickup_effects));

    string obs_num_var_pre_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                  + "obs_num_"
                                  + std::to_string(step - 1);
    z3::expr obs_num_var_pre = ctx.int_const(obs_num_var_pre_name.c_str());
    string obs_num_var_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                + "obs_num_"
                                + std::to_string(step);
    z3::expr obs_num_var_k = ctx.int_const(obs_num_var_k_name.c_str());
    pick_up_trans.push_back(obs_num_var_k == obs_num_var_pre);

    pick_up_trans.push_back(genPickUpSafeCond(ctx, step));
    return mk_and(pick_up_trans);
}

void CorridorDomain::updateGroupSplitIdsForAction(int action_idx,
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


void CorridorDomain::updateBeliefGroupMap(int step) {

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

void CorridorDomain::updateGroupSplitIds(unordered_map<int, string> &group_split_ids, int step) {
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

z3::expr CorridorDomain::genTransCond(z3::context &ctx, int step) {
    z3::expr_vector trans_conds(ctx);
    if (step > belief_group_size.size()) {
        MyUtils::printErrorMsg("incorrect belief group map update");
    }
    if (step == belief_group_size.size()) {
        updateBeliefGroupMap(step);
    }

    if (MyUtils::solver_name == "partial") {
        if (a_start == look_north_idx) {
            if (o_start == positive_observation) {
                if (step == start_step + 1) {
                    trans_conds.push_back(genMoveActionTransCond(ctx, move_west_idx, step));
                    return mk_or(trans_conds);
                } else if (step == start_step + 2) {
                    trans_conds.push_back(genMoveActionTransCond(ctx, move_north_idx, step));
                    return mk_or(trans_conds);
                } else if (step == start_step + 3) {
                    trans_conds.push_back(genMoveActionTransCond(ctx, move_north_idx, step));
                    return mk_or(trans_conds);
                }
            }
        }
    }
    trans_conds.push_back(genMoveActionTransCond(ctx, move_north_idx,step));
    //trans_conds.push_back(genMoveActionTransCond(ctx, move_south_idx,step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_east_idx,step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_west_idx, step));
    if (see_obs_num < num_obs) {
        if (MyUtils::solver_mode != "success") {
            trans_conds.push_back(genLookActionTransCond(ctx, look_north_idx, step));
        }
        //trans_conds.push_back(genLookActionTransCond(ctx, look_south_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_east_idx, step));
        //trans_conds.push_back(genLookActionTransCond(ctx, look_west_idx, step));
    }
    trans_conds.push_back(genPickUpTransCond(ctx, pick_up_left_hand_idx, step));
    trans_conds.push_back(genPickUpTransCond(ctx, pick_up_right_hand_idx, step));
    return mk_or(trans_conds);
}

z3::expr CorridorDomain::genMoveSafeCond(z3::context &ctx, int step) {
    z3::expr_vector unsafe_states(ctx);
    auto &b_to_g_map = belief_to_group_maps[step];
    vector<int> unsafe_g;
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        unsafe_g.push_back(0);
    }

    for (auto b_to_g : b_to_g_map) {
        int idx = b_to_g.first;
        int r = getRobotLocIdx(idx);
        int i = getObsLocsIdx(idx);

        if (in_collision(r, obs_locs_list[i])) {
            int g_idx = b_to_g_map[idx];
            ++ unsafe_g[g_idx];
        }
    }

    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        if (unsafe_g[g] == 0) {
            continue;
        }
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        unsafe_states.push_back(b_g_k * unsafe_g[g]);
    }
    if (unsafe_states.size() == 0) {
        return ctx.bool_val(true);
    }
    return sum(unsafe_states) <
            ctx.real_val(to_string(delta).c_str()) * genBeliefSum(ctx, step);
}

z3::expr CorridorDomain::genPickUpSafeCond(z3::context &ctx, int step) {
    z3::expr_vector goal_states(ctx);
    goal_states.push_back(ctx.int_val(0));
    z3::expr_vector unsafe_states(ctx);
    z3::expr_vector ready_states(ctx);
    ready_states.push_back(ctx.int_val(0));
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
    if (unsafe_states.size() == 0) {
        return ctx.bool_val(true);
    }
    return sum(unsafe_states) <
            ctx.real_val(to_string(0.2).c_str())
            * (sum(goal_states) + sum(unsafe_states) + sum(ready_states));
}


z3::expr CorridorDomain::genGoalCond(z3::context &ctx, int step) {
    z3::expr_vector goal_states(ctx);
    z3::expr_vector unsafe_states(ctx);
    unsafe_states.push_back(ctx.int_val(0));
    z3::expr_vector ready_states(ctx);
    ready_states.push_back(ctx.int_val(0));
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
    string obs_num_var_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                + "obs_num_"
                                + std::to_string(step);
    z3::expr obs_num_var_k = ctx.int_const(obs_num_var_k_name.c_str());
    goal_conds.push_back(obs_num_var_k <= ctx.int_val(num_obs));

    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    goal_conds.push_back((a_k == ctx.int_val(pick_up_left_hand_idx))
                         || (a_k == ctx.int_val(pick_up_right_hand_idx)));

    return mk_and(goal_conds);
}

vector<pair<int, double>> CorridorDomain::getObservationDistribution(
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

string CorridorDomain::getActionMeaning(int action_id) {
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

string CorridorDomain::getObservationMeaning(int observation_id) {
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

pair<vector<double>, vector<double>> CorridorDomain::extractProbs(BeliefStatePtr b, int step) {
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


string CorridorDomain::beliefStateToString(BeliefStatePtr b, int step) {

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


vector<z3::expr> CorridorDomain::getBeliefStateVarList(z3::context &ctx, int step) {

    vector<z3::expr> b;
    for (int g = 0; g < belief_group_size[step].size(); ++g) {
        std::string b_g_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(g) + "_"
                                 + std::to_string(step);
        z3::expr b_g_k = ctx.real_const(b_g_k_name.c_str());
        b.push_back(b_g_k);
    }
    return b;
}

BeliefStatePtr CorridorDomain::getBeliefState(z3::context &ctx, ModelPtr model, int step) {
    // TODO get the belief from z3
    vector<z3::expr> b_vars = getBeliefStateVarList(ctx, step);
    vector<double> b_vals;
    double b_sum = 0.0;
    for (int i = 0; i < b_vars.size(); ++i) {
        z3::expr b_i = model->eval(b_vars[i]);
        double n = stod(b_i.numerator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
        double d = stod(b_i.denominator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
        b_vals.push_back(n / d);
        b_sum += (n / d) * belief_group_size[step][i];
    }
    for (int i = 0; i < b_vals.size(); ++i) {
        b_vals[i] = b_vals[i] / b_sum;
    }
    return make_shared <BeliefState> (b_vals);
}

int CorridorDomain::observe(BeliefStatePtr b, int action_id, int step) {
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

void CorridorDomain::generateRandomInstances(string test_file_dir, int num) {
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

void CorridorDomain::computeAlphaVector(PolicyNodePtr policy) {
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

double CorridorDomain::computeBeliefValue(BeliefStatePtr b, unordered_map<int, double> &alpha, int step) {
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

unordered_map<int, bool> CorridorDomain::getAvailableActions() {
    unordered_map<int, bool> actions;
    actions[move_north_idx] = true;
    //actions[move_south_idx] = true;
    actions[move_west_idx] = true;
    actions[move_east_idx] = true;
    actions[look_north_idx] = true;
    // actions[look_south_idx] = true;
    // actions[look_west_idx] = true;
    actions[look_east_idx] = true;
    actions[pick_up_left_hand_idx] = true;
    actions[pick_up_right_hand_idx] = true;
    return actions;
};

BeliefStatePtr CorridorDomain::getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
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

        vector<double> move_effects;
        vector<bool> has_pre;
        vector<bool> has_next;

        // initialize
        for (int g = 0; g < belief_group_size[step].size(); ++g) {
            move_effects.push_back(0.0);
            has_pre.push_back(false);
            has_next.push_back(false);
        }

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

        vector<double> b_next_vals;
        for (int g = 0; g < belief_group_size[step].size(); ++g) {
            b_next_vals.push_back(move_effects[g]);
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

bool CorridorDomain::isGoal(BeliefStatePtr b, int step) {
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

#endif
