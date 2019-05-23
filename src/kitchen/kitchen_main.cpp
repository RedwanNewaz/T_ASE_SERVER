//
//
// Created by Redwan Newaz on 2019-04-27.
//

#include "kitchen/kitchen_main.h"
#include "kitchen/kitchen_tau.h"
#include "my_utils.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_set>

using namespace std;


kitchen_main::kitchen_main(string test_file_path, bool run_init) {
    // read test description
#ifdef VIZ_SYNTHESIS
    std::unique_lock<mutex>lk(SHARE->MU);
#endif
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
#ifdef VIZ_SYNTHESIS
            SHARE->gt_obs.push_back(QPoint(x,y));
#endif
        }
        test_file >> epsilon;
        test_file.close();
#ifdef VIZ_SYNTHESIS
        SHARE->set(x_size, y_size);
#endif



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
#ifdef VIZ_SYNTHESIS
    SHARE->obstacle_size = obs_locs_list.size();
    SHARE->map_status = true;
    lk.unlock();
    SHARE->COND.notify_all();
#endif

}

kitchen_main::kitchen_main(int s, BeliefStatePtr init_b,
                               int a_s, int o_s,
                               unordered_map<int, int> init_b_to_g_map, bool run_init):
        kitchen_belief(s, init_b, a_s, o_s, init_b_to_g_map, run_init) {

}

POMDPDomainPtr kitchen_main::createNewPOMDPDomain(int s, BeliefStatePtr new_init_b,
                                                    int a_s, int o_s) {


    return make_shared <kitchen_main> (s, filterZeroProb(new_init_b, s),
                                         a_s, o_s, belief_to_group_maps[s]);
}

z3::expr kitchen_main::genInitCond(z3::context &ctx) {
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

z3::expr kitchen_main::genBeliefSum(z3::context &ctx, int step) {
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



z3::expr kitchen_main::genMoveActionPreCond(z3::context &ctx, int action_idx, int step) {
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

z3::expr kitchen_main::genMoveActionTransCond(z3::context &ctx, int action_idx, int step) {
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

z3::expr kitchen_main::genLookActionPreCond(z3::context &ctx, int action_idx, int step) {
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

z3::expr kitchen_main::genLookActionTransCond(z3::context &ctx, int action_idx, int step) {
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

z3::expr kitchen_main::genPickUpTransCond(z3::context &ctx, int action_idx, int step) {
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


z3::expr kitchen_main::genTransCond(z3::context &ctx, int step) {
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
    trans_conds.push_back(genMoveActionTransCond(ctx, move_south_idx,step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_east_idx,step));
    trans_conds.push_back(genMoveActionTransCond(ctx, move_west_idx, step));
    if (see_obs_num < num_obs) {
        if (MyUtils::solver_mode != "success") {
            trans_conds.push_back(genLookActionTransCond(ctx, look_north_idx, step));
        }
        trans_conds.push_back(genLookActionTransCond(ctx, look_south_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_east_idx, step));
        trans_conds.push_back(genLookActionTransCond(ctx, look_west_idx, step));
    }
    trans_conds.push_back(genPickUpTransCond(ctx, pick_up_left_hand_idx, step));
    trans_conds.push_back(genPickUpTransCond(ctx, pick_up_right_hand_idx, step));
    return mk_or(trans_conds);
}

z3::expr kitchen_main::genMoveSafeCond(z3::context &ctx, int step) {
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

z3::expr kitchen_main::genPickUpSafeCond(z3::context &ctx, int step) {
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


z3::expr kitchen_main::genGoalCond(z3::context &ctx, int step) {
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


vector<z3::expr> kitchen_main::getBeliefStateVarList(z3::context &ctx, int step) {

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

BeliefStatePtr kitchen_main::getBeliefState(z3::context &ctx, ModelPtr model, int step) {
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

BeliefStatePtr kitchen_main::getNextBeliefDirect(BeliefStatePtr b_pre, int action_id, int obs_id, int step) {

    auto belief_next = kitchen_belief::getNextBeliefDirect(b_pre, action_id, obs_id, step);

    if(belief_next != nullptr){
        kitchen_tau tau(this, b_pre, action_id, obs_id, step);
        cout<<"computing tau"<<endl;
        auto res = tau.compute();
    }
    return belief_next;
}

vector<pair<int, double>> kitchen_main::getObservationDistribution(int action_id, BeliefStatePtr b_pre, int step) {
    kitchen_tau tau(this, b_pre, action_id, 0, step);
    auto obs_next = tau.pdfZ();
    return obs_next;
}

void kitchen_main::publish(BeliefStatePtr b, int step) {
    /**@copybrief
     * currently timer is synchronize the visualization thread \par
     * I wanna synchronize thread using conditional variable defined in SharedData class \par
     *
     */
#ifdef VIZ_SYNTHESIS



    auto probs = extractProbs(b, step);
    auto &b_r_vals = probs.first;
    auto &b_obs_vals = probs.second;


    vector<double> obs_probs(obs_all_locs.size(),0.0);




    std::unique_lock<mutex>lk(SHARE->MU);
    SHARE->reset_map();
    SHARE->step = step;
// STEP - compute robot probability
    double max_prob = -numeric_limits<double>::max();
    int robot_index = 0;
    for (int r = 0; r < rob_locs_size; ++r) {

        double rob_prob = b_r_vals[r];
        if(rob_prob>max_prob)
        {
            max_prob = rob_prob;
            robot_index = r;
        }
    }

    // STEP - compute obstacle probability
    double obs_prob = -numeric_limits<double>::max();
    int obs_index = 0;
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        double obs_i_prob = b_obs_vals[i];
        if (obs_i_prob == 0) {
            continue;
        }
        for (auto obs_loc : obs_locs_list[i]) {
            int idx = obs_loc - y_size;
            assert(idx<obs_all_locs.size());
            obs_probs[idx] += obs_i_prob;
            if(obs_probs[idx]>obs_prob)
            {
                obs_prob = obs_probs[idx];
                obs_index = obs_loc;
            }

        }
    }

    auto point = [&](int index ){
        auto x = getXY(index);
        return QPoint(x.first,x.second);};


    qDebug()<<max_prob<<point(robot_index)<<obs_prob<<point(obs_index);

    SHARE->map.push_back(make_pair(max_prob,point(robot_index)));
//    SHARE->obstacles.push_back(make_pair(obs_prob,point(obs_index)));

    map<double,QPoint> obstacle_map;
    for (int i = 0; i < obs_locs_list.size(); ++i) {
        double obs_i_prob = b_obs_vals[i];
        if (obs_i_prob == 0) {
            continue;
        }
        for (auto obs_loc : obs_locs_list[i]) {
            int idx = obs_loc - y_size;
            assert(idx<obs_all_locs.size());
            obstacle_map[obs_probs[idx]]= point(obs_loc);
        }
    }
    int N = 0;
    for(auto it=obstacle_map.begin();it!=obstacle_map.end();++it)
    {
        SHARE->obstacles.push_back(make_pair(it->first,it->second));
        if(N++>SHARE->gt_obs.size())
            break;
    }




    SHARE->map_status = true;
    lk.unlock();
    SHARE->COND.notify_all();
#endif
}

kitchen_main::~kitchen_main() {
    /*
     * by default finish value is false.
     * change SHARE->finish  to true to terminate awaiting visualization thread
     */
#ifdef VIZ_SYNTHESIS
    std::unique_lock<mutex>lk(SHARE->MU);
        SHARE->map_status = true;
        SHARE->finish = true;
    lk.unlock();
    SHARE->COND.notify_all();
#endif
}


