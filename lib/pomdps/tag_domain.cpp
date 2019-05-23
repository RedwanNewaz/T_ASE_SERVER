//
// Created by Yue Wang on 1/20/18.
//

#include "my_utils.h"
#include "partial_policy.h"
#include "pomdps/tag_domain.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>

int TagDomain::curr_target_loc = 0;
double TagDomain::tag_th = 0.0;

/*
 * Tag: robot an target move simultaneously. Target move according to robot's current
 * position (not next position). Robot's observation based on target's distribution after
 * moving
 */

TagDomain::TagDomain(string test_file_path): POMDPDomain(0, 0, 0), rob_locs_set() {
    // read test description
    ifstream test_file;
    test_file.open(test_file_path);
    if (test_file.is_open()) {
        test_file >> tag_th;
        test_file >> rob_init_loc;
        test_file >> curr_target_loc;
        test_file >> epsilon;
    }
    test_file.close();
    MyUtils::printDebugMsg("tag threshold: " + to_string(tag_th),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("initial robot location: " + to_string(rob_init_loc),  MyUtils::LEVEL_ONE_MSG);
    MyUtils::printDebugMsg("initial target location: " + to_string(curr_target_loc),  MyUtils::LEVEL_ONE_MSG);

    vector<pair<double, double>> init_b_vals;
    // uniform distribution of target location
    for (int i = 0; i < belief_size; ++i) {
        init_b_vals.push_back(make_pair(1, belief_size));
    }


    // tagged state - false
    init_b_vals.push_back(make_pair(0, 1));

    // robot location
    init_b_vals.push_back(make_pair(rob_init_loc, 1));

    // b_sum_pre
    init_b_vals.push_back(make_pair(1, 1));

    rob_locs_set.push_back(unordered_map<int, bool> ());
    for (int r = 0; r < belief_size; ++r) {
        rob_locs_set[0][r] = false;
    }
    rob_locs_set[0][rob_init_loc] = true;

    init_belief = make_shared <BeliefState> (init_b_vals);
}

TagDomain::TagDomain(int s, BeliefStatePtr init_b, int a_s, int o_s):
        POMDPDomain(s, a_s, o_s), rob_locs_set() {
    auto b_vals = init_b->getBeliefStateExactValues();
    double b_sum = 0.0;
    for (int i = 0; i < belief_size; ++i) {
        double val = b_vals[i].first / b_vals[i].second;
        if (val < zero_th) {
            b_vals[i].first = 0;
            b_vals[i].second = 1;
        } else {
            b_vals[i].first  = int(val * MyUtils::DOUBLE_ROUND_FACTOR);
            b_vals[i].second = MyUtils::DOUBLE_ROUND_FACTOR;
        }
        b_sum += b_vals[i].first / b_vals[i].second;
    }
    // normalize
    for (int i = 0; i < belief_size; ++i) {
        double val = b_vals[i].first / b_vals[i].second / b_sum;
        b_vals[i].first  = int(val * MyUtils::DOUBLE_ROUND_FACTOR);
        b_vals[i].second = MyUtils::DOUBLE_ROUND_FACTOR;
    }

    // normalize tag state
    double b_sum_pre = b_vals[belief_size + 2].first / b_vals[belief_size + 2].second;
    double val = b_vals[belief_size].first / b_vals[belief_size].second / b_sum_pre;
    b_vals[belief_size].first  = int(val * MyUtils::DOUBLE_ROUND_FACTOR);
    b_vals[belief_size].second = MyUtils::DOUBLE_ROUND_FACTOR;


    for (int k = 0; k <= s; ++k) {
        rob_locs_set.push_back(unordered_map<int, bool> ());
    }
    for (int r = 0; r < belief_size; ++r) {
        rob_locs_set[s][r] = false;
    }
    rob_init_loc = b_vals[belief_size + 1].first / b_vals[belief_size + 1].second;
    rob_locs_set[s][rob_init_loc] = true;

    init_belief = make_shared <BeliefState> (b_vals);
    MyUtils::printDebugMsg("init belief: " + beliefStateToString(init_belief, start_step),
                           MyUtils::LEVEL_ONE_MSG);
}

POMDPDomainPtr TagDomain::createNewPOMDPDomain(int s, BeliefStatePtr init_b,
                                    int a_s, int o_s) {

    return make_shared <TagDomain> (s, init_b, a_s, o_s);
}

z3::expr TagDomain::genInitCond(z3::context &ctx) {

    z3::expr_vector init_conds(ctx);

    vector<z3::expr> b_s_vars = getBeliefStateVarList(ctx, start_step);
    auto b_s_vals = init_belief->getBeliefStateExactValues();
    auto b_s_exprs = init_belief->getBeliefStateExprs(ctx);
    if (b_s_vars.size() != b_s_vals.size()) {
        MyUtils::printErrorMsg(
                "when generating initial condition, the sizes of belief vars and belief values do not agree"
        );
    }
    for (int i = 0; i < b_s_vals.size(); ++i) {
        if (b_s_vars[i].is_int()) {
            init_conds.push_back(b_s_vars[i] == ctx.int_val(int(b_s_vals[i].first)));
        } else {
            init_conds.push_back(b_s_vars[i] == b_s_exprs[i]);
        }
    }
    return mk_and(init_conds);
}

int TagDomain::getX(int idx) {
    if (idx <= 19) {
        return idx % 10;
    }
    return 5 + (idx - 20) % 3;
}

int TagDomain::getY(int idx) {
    if (idx <= 19) {
        return floor(idx / 10);
    }
    return 2 + floor((idx - 20) / 3);
}

int TagDomain::distance(int l1, int l2) {
    int x1 = getX(l1);
    int y1 = getY(l1);

    int x2 = getX(l2);
    int y2 = getY(l2);
    return abs(x1 - x2) + abs(y1 - y2);
}

int TagDomain::getNextLoc(int action_id, int curr_loc) {

    // if can not move, will stay
    int next_loc = curr_loc;
    if (action_id == north_idx) {
        if ((curr_loc >= 0) && (curr_loc <= 9)) {
            next_loc = curr_loc + 10;
        } else if ((curr_loc >= 15) && (curr_loc <= 17)) {
            next_loc = curr_loc + 5;
        } else if ((curr_loc >= 20) && (curr_loc <= 25)) {
            next_loc = curr_loc + 3;
        }
    } else if (action_id == south_idx) {
        if ((curr_loc >= 10) && (curr_loc <= 19)) {
            next_loc = curr_loc - 10;
        } else if ((curr_loc >= 20) && (curr_loc <= 22)) {
            next_loc = curr_loc - 5;
        } else if ((curr_loc >= 23) && (curr_loc <= 28)) {
            next_loc = curr_loc - 3;
        }
    } else if (action_id == west_idx) {
        if (!((curr_loc == 0) || (curr_loc == 10) ||
              (curr_loc == 20) || (curr_loc == 23) ||
              (curr_loc == 26))) {
            next_loc = curr_loc - 1;
        }
    } else if (action_id == east_idx) {
        if (!((curr_loc == 9) || (curr_loc == 19) ||
              (curr_loc == 22) || (curr_loc == 25) ||
              (curr_loc == 28))) {
            next_loc = curr_loc + 1;
        }
    }
    return next_loc;
}

vector<double> TagDomain::getNextTargetLocDist(int r_loc, int t_loc) {

    int rob_pos_x = getX(r_loc);
    int rob_pos_y = getY(r_loc);

    vector<double> next_target_b;
    for (int i = 0; i < belief_size; ++i) {
        next_target_b.push_back(0.0);
    }

    next_target_b[t_loc] += 0.2;
    int tar_pos_x = getX(t_loc);
    int tar_pos_y = getY(t_loc);
    if (tar_pos_x == rob_pos_x) {
        // move west: 0.2
        next_target_b[getNextLoc(west_idx, t_loc)] += 0.2;
        // move east: 0.2
        next_target_b[getNextLoc(east_idx, t_loc)] += 0.2;
    } else if (tar_pos_x < rob_pos_x) {
        // move west: 0.4
        next_target_b[getNextLoc(west_idx, t_loc)] += 0.4;
    } else if (tar_pos_x > rob_pos_x) {
        // move east: 0.4
        next_target_b[getNextLoc(east_idx, t_loc)] += 0.4;
    }
    if (tar_pos_y == rob_pos_y) {
        // move north: 0.2
        next_target_b[getNextLoc(north_idx, t_loc)] += 0.2;
        // move south: 0.2
        next_target_b[getNextLoc(south_idx, t_loc)] += 0.2;
    } else if (tar_pos_y > rob_pos_y) {
        // move north: 0.4
        next_target_b[getNextLoc(north_idx, t_loc)] += 0.4;
    } else if (tar_pos_y < rob_pos_y) {
        // move south: 0.4
        next_target_b[getNextLoc(south_idx, t_loc)] += 0.4;
    }
    return next_target_b;
}

z3::expr TagDomain::genGoalCond(z3::context &ctx, int step) {
    // tagged state
    if (step == 0) {
        return ctx.bool_val(false);
    }
    std::string b_k_tagged_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                  + std::to_string(step) + "_tagged";
    z3::expr b_k_tagged = ctx.real_const(b_k_tagged_name.c_str());

    std::string b_sum_pre_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step) + "_sum_pre";
    z3::expr b_sum_pre_k = ctx.real_const(b_sum_pre_k_name.c_str());

    return (b_k_tagged >= ctx.real_val(to_string(tag_th).c_str()) * b_sum_pre_k);
}

z3::expr TagDomain::genBeliefNoChange(z3::context &ctx, int step) {
    z3::expr_vector belief_no_change(ctx);
    for (int i = 0; i < 29; ++i) {
        std::string b_pre_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step - 1) + "_"
                                   + std::to_string(i);
        z3::expr b_pre_i = ctx.real_const(b_pre_i_name.c_str());
        std::string b_k_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(step) + "_"
                                 + std::to_string(i);
        z3::expr b_k_i = ctx.real_const(b_k_i_name.c_str());
        belief_no_change.push_back(b_k_i == b_pre_i);
    }
    return mk_and(belief_no_change);
}

z3::expr TagDomain::genBeliefSum(z3::context &ctx, int step) {
    z3::expr_vector b_sum(ctx);
    for (int i = 0; i < 29; ++i) {
        std::string b_k_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(step) + "_"
                                 + std::to_string(i);
        z3::expr b_k_i = ctx.real_const(b_k_i_name.c_str());
        b_sum.push_back(b_k_i);
    }
    return sum(b_sum);
}

z3::expr TagDomain::genMoveTransCond(z3::context &ctx, int action_id, int step) {
    z3::expr_vector move_trans(ctx);
    // a_k = action_id, ;
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    move_trans.push_back(a_k == ctx.int_val(action_id));

    // pre cond: not goal
    move_trans.push_back(!genGoalCond(ctx, step - 1));

    z3::expr_vector move_effects(ctx);

    string rob_pos_pre_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 1);
    z3::expr rob_pos_pre = ctx.int_const(rob_pos_pre_name.c_str());
    string rob_pos_k_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step);
    z3::expr rob_pos_k = ctx.int_const(rob_pos_k_name.c_str());

    auto &rob_locs_set_pre = rob_locs_set[step - 1];
    z3::expr b_sum_pre = genBeliefSum(ctx, step - 1);
    for (int r = 0; r < belief_size; ++r) {
        if (!rob_locs_set_pre[r]) {
            continue;
        }
        // move to next loc
        int r_next = getNextLoc(action_id, r);

        if (r == r_next) {
            continue;
        }

        z3::expr_vector effects_r(ctx);
        effects_r.push_back(rob_pos_k == r_next);


        effects_r.push_back(rob_pos_pre == r);

        // pre cond: not in the same cell
        std::string b_pre_r_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step - 1) + "_"
                                   + std::to_string(r);
        z3::expr b_pre_r = ctx.real_const(b_pre_r_name.c_str());
        effects_r.push_back(!(b_pre_r == 1));

        // two observations
        z3::expr_vector yes_obs_conds(ctx);
        z3::expr_vector no_obs_conds(ctx);

        string o_k_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
        z3::expr o_k = ctx.int_const(o_k_name.c_str());
        yes_obs_conds.push_back(o_k == ctx.int_val(yes_obs));
        no_obs_conds.push_back(o_k == ctx.int_val(no_obs));

        z3::expr_vector yes_probs(ctx);
        for (int i = 0; i < belief_size; ++i) {
            z3::expr i_expr = ctx.int_val(i);
            std::string b_pre_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                       + std::to_string(step - 1) + "_"
                                       + std::to_string(i);
            z3::expr b_pre_i = ctx.real_const(b_pre_i_name.c_str());

            vector<double> next_target_dist = getNextTargetLocDist(r, i);
            if (next_target_dist[r_next] > 0) {
                z3::expr t_prob = ctx.real_val(to_string(next_target_dist[r_next]).c_str());
                yes_probs.push_back(b_pre_i * t_prob);
            }
        }
        // for yes: probability > threshold
        if (MyUtils::solver_name == "partial") {
            yes_obs_conds.push_back(sum(yes_probs) > ctx.real_val(to_string(yes_th).c_str()) * b_sum_pre);
        } else {
            effects_r.push_back(sum(yes_probs) > ctx.real_val(to_string(yes_th).c_str()) * b_sum_pre);
        }

        // for no: probability < threshold
        no_obs_conds.push_back(sum(yes_probs) < ctx.real_val(to_string(1 - yes_th).c_str()) * b_sum_pre);

        // update tagged state
        std::string b_k_tagged_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                      + std::to_string(step) + "_tagged";
        z3::expr b_k_tagged = ctx.real_const(b_k_tagged_name.c_str());
        yes_obs_conds.push_back(b_k_tagged == 0);
        no_obs_conds.push_back(b_k_tagged == sum(yes_probs));


        // update target dist
        for (int j = 0; j < belief_size; ++j) {
            std::string b_k_j_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                     + std::to_string(step) + "_"
                                     + std::to_string(j);
            z3::expr b_k_j = ctx.real_const(b_k_j_name.c_str());

            if (r_next == j) {
                yes_obs_conds.push_back(b_k_j == 1);
                no_obs_conds.push_back(b_k_j == 0);
            } else {
                yes_obs_conds.push_back(b_k_j == 0);
                z3::expr_vector j_probs(ctx);
                for (int i = 0; i < belief_size; ++i) {
                    z3::expr i_expr = ctx.int_val(i);
                    std::string b_pre_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                               + std::to_string(step - 1) + "_"
                                               + std::to_string(i);
                    z3::expr b_pre_i = ctx.real_const(b_pre_i_name.c_str());
                    vector<double> next_target_dist = getNextTargetLocDist(r, i);
                    if (next_target_dist[j] > 0) {
                        z3::expr t_prob = ctx.real_val(to_string(next_target_dist[j]).c_str());
                        j_probs.push_back(b_pre_i * t_prob);
                    }
                }
                no_obs_conds.push_back(ite(sum(j_probs) > ctx.real_val(to_string(zero_th).c_str()) * b_sum_pre,
                                           b_k_j == sum(j_probs), b_k_j == 0
                                       ));
            }
        }
        effects_r.push_back(mk_and(yes_obs_conds) || mk_and(no_obs_conds));
        move_effects.push_back(mk_and(effects_r));
    }
    move_trans.push_back(mk_or(move_effects));
    if (MyUtils::solver_name == "partial") {
        if (step > 2) {
            string rob_pos_pre_pre_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 2);
            z3::expr rob_pos_pre_pre = ctx.int_const(rob_pos_pre_pre_name.c_str());
            move_trans.push_back(!(rob_pos_k == rob_pos_pre_pre));
        } else if (step > 4) {
            string rob_pos_pre_loop_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 4);
            z3::expr rob_pos_pre_loop = ctx.int_const(rob_pos_pre_loop_name.c_str());
            move_trans.push_back(!(rob_pos_k == rob_pos_pre_loop));
        }
    }
    return mk_and(move_trans);
}

z3::expr TagDomain::genTagTransCond(z3::context &ctx, int step) {
    z3::expr_vector tag_trans(ctx);
    // a_k = tag_idx;
    string a_k_name = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr a_k = ctx.int_const(a_k_name.c_str());
    tag_trans.push_back(a_k == ctx.int_val(tag_idx));

    // pre cond: not goal
    tag_trans.push_back(!genGoalCond(ctx, step - 1));
    z3::expr_vector move_effects(ctx);

    string rob_pos_pre_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step - 1);
    z3::expr rob_pos_pre = ctx.int_const(rob_pos_pre_name.c_str());
    string rob_pos_k_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step);
    z3::expr rob_pos_k = ctx.int_const(rob_pos_k_name.c_str());

    // robot not moving
    tag_trans.push_back(rob_pos_k == rob_pos_pre);

    // no change in target belief
    tag_trans.push_back(genBeliefNoChange(ctx, step));

    // observation: tag
    string o_k_name = MyUtils::OBSERVATION_VAR_PREFIX + "_" + std::to_string(step);
    z3::expr o_k = ctx.int_const(o_k_name.c_str());

    tag_trans.push_back(o_k == ctx.int_val(tag_success_obs));

    z3::expr_vector tag_effects(ctx);
    for (int r = 0; r < belief_size; ++r) {
        z3::expr_vector effects_r(ctx);

        effects_r.push_back(rob_pos_pre == r);

        // pre cond: in the same cell
        std::string b_pre_r_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step - 1) + "_"
                                   + std::to_string(r);
        z3::expr b_pre_r = ctx.real_const(b_pre_r_name.c_str());
        effects_r.push_back((b_pre_r == 1));

        // effect: tag
        std::string b_k_tagged_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                      + std::to_string(step) + "_tagged";
        z3::expr b_k_tagged = ctx.real_const(b_k_tagged_name.c_str());
        effects_r.push_back(b_k_tagged == 1);

        tag_effects.push_back(mk_and(effects_r));
    }

    tag_trans.push_back(mk_or(tag_effects));

    return mk_and(tag_trans);
}

z3::expr TagDomain::genTransCond(z3::context &ctx, int step) {
    z3::expr_vector trans_conds(ctx);

    // update robot loc set
    if (step > rob_locs_set.size()) {
        MyUtils::printErrorMsg("incorrect robot location update");
    }
    if (step == rob_locs_set.size()) {
        rob_locs_set.push_back(unordered_map<int, bool> ());
        for (int r = 0; r < belief_size; ++r) {
            rob_locs_set[step][r] = false;
        }

        for (int r = 0; r < belief_size; ++r) {
            if (rob_locs_set[step - 1][r]) {
                int next_loc = getNextLoc(north_idx, r);
                if (next_loc != r) {
                    rob_locs_set[step][next_loc] = true;
                }
                next_loc = getNextLoc(south_idx, r);
                if (next_loc != r) {
                    rob_locs_set[step][next_loc] = true;
                }
                next_loc = getNextLoc(west_idx, r);
                if (next_loc != r) {
                    rob_locs_set[step][next_loc] = true;
                }
                next_loc = getNextLoc(east_idx, r);
                if (next_loc != r) {
                    rob_locs_set[step][next_loc] = true;
                }
                // rob_locs_set[step][getNextLoc(stay_idx, r)] = true;
            }
        }
    }

    if (MyUtils::solver_name == "partial") {
        if (step == start_step + 1) {
            int next_loc = getNextLoc(north_idx, rob_init_loc);
            if (next_loc != rob_init_loc) {
                trans_conds.push_back(genMoveTransCond(ctx, north_idx, step));
            }
            next_loc = getNextLoc(south_idx, rob_init_loc);
            if (next_loc != rob_init_loc) {
                trans_conds.push_back(genMoveTransCond(ctx, south_idx, step));
            }
            next_loc = getNextLoc(west_idx, rob_init_loc);
            if (next_loc != rob_init_loc) {
                trans_conds.push_back(genMoveTransCond(ctx, west_idx, step));
            }
            next_loc = getNextLoc(east_idx, rob_init_loc);
            if (next_loc != rob_init_loc) {
                trans_conds.push_back(genMoveTransCond(ctx, east_idx, step));
            }
            trans_conds.push_back(genTagTransCond(ctx, step));

            std::string b_sum_pre_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                           + std::to_string(step) + "_sum_pre";
            z3::expr b_sum_pre_k = ctx.real_const(b_sum_pre_k_name.c_str());

            return mk_or(trans_conds) && (b_sum_pre_k == genBeliefSum(ctx, step - 1));
        }
    }

    trans_conds.push_back(genMoveTransCond(ctx, north_idx, step));
    trans_conds.push_back(genMoveTransCond(ctx, south_idx, step));
    trans_conds.push_back(genMoveTransCond(ctx, west_idx, step));
    trans_conds.push_back(genMoveTransCond(ctx, east_idx, step));
    // trans_conds.push_back(genMoveTransCond(ctx, stay_idx, step));
    trans_conds.push_back(genTagTransCond(ctx, step));

    std::string b_sum_pre_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step) + "_sum_pre";
    z3::expr b_sum_pre_k = ctx.real_const(b_sum_pre_k_name.c_str());

    return mk_or(trans_conds) && (b_sum_pre_k == genBeliefSum(ctx, step - 1));

}

vector<pair<int, double>> TagDomain::getObservationDistribution(
        int action_id, BeliefStatePtr b_pre, int step) {
    vector<pair<int, double>> observations;
     if (action_id == north_idx || action_id == south_idx
         || action_id == west_idx || action_id == east_idx || action_id == stay_idx) {
         vector<double> b_pre_vals = b_pre->getBeliefStateValues();
         int curr_rob_loc = b_pre_vals[belief_size + 1];
         int next_rob_loc = getNextLoc(action_id, curr_rob_loc);

         double yes_prob = 0.0;

         double b_sum_pre = 0.0;
         for (int i = 0; i < belief_size; ++i) {
             vector<double> next_target_loc_dist = getNextTargetLocDist(curr_rob_loc, i);
             yes_prob += next_target_loc_dist[next_rob_loc] * b_pre_vals[i];
             b_sum_pre += b_pre_vals[i];
         }

         yes_prob = yes_prob / b_sum_pre;

         if (yes_prob > yes_th) {
             observations.push_back(make_pair(yes_obs, yes_prob));
         }
         if (yes_prob < 1 - yes_th ) {
             observations.push_back(make_pair(no_obs, 1 - yes_prob));
         }
    } else if (action_id == tag_idx) {
        observations.push_back(make_pair (tag_success_obs, 1.0));
    }
    return observations;
};

string TagDomain::beliefStateToString(BeliefStatePtr b, int step) {
    std::stringstream ss;
    vector<double> b_vals = b->getBeliefStateValues();

    string b_k_str = "[ ";
    for (int i = 0; i < belief_size; ++i) {
        b_k_str += to_string(i) + ": " + to_string(b_vals[i]) + ", ";
    }

    b_k_str += "]";
    ss << "target belief: " << b_k_str << endl;

    ss << "target tagged: " << b_vals[belief_size] << endl;

    double rob_loc = b_vals[belief_size + 1];
    ss << "robot pose: " << rob_loc << endl;

    ss << "previous belief sum: " << b_vals[belief_size + 2] << endl;

    return ss.str();
}

vector<z3::expr> TagDomain::getBeliefStateVarList(z3::context &ctx, int step){
    vector<z3::expr> b;

    for (int i = 0; i < belief_size; ++i) {
        std::string b_k_i_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                 + std::to_string(step) + "_"
                                 + std::to_string(i);
        z3::expr b_k_i = ctx.real_const(b_k_i_name.c_str());
        b.push_back(b_k_i);
    }

    std::string b_k_tagged_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                  + std::to_string(step) + "_tagged";
    z3::expr b_k_tagged = ctx.real_const(b_k_tagged_name.c_str());
    b.push_back(b_k_tagged);

    string rob_pos_k_name = MyUtils::ROB_POS_VAR + "_" + std::to_string(step);
    z3::expr rob_pos_k = ctx.int_const(rob_pos_k_name.c_str());
    b.push_back(rob_pos_k);

    std::string b_sum_pre_k_name = MyUtils::BELIEF_VAR_PREFIX + "_"
                                   + std::to_string(step) + "_sum_pre";
    z3::expr b_sum_pre_k = ctx.real_const(b_sum_pre_k_name.c_str());
    b.push_back(b_sum_pre_k);

    return b;
}

BeliefStatePtr TagDomain::getBeliefState(z3::context &ctx, ModelPtr model, int step)  {

    vector<z3::expr> b_vars = getBeliefStateVarList(ctx, step);
    vector<pair<double, double>> b_vals;

    double b_sum = 0.0;
    for (int i = 0; i < belief_size; ++i) {
        z3::expr b_i = model->eval(b_vars[i]);
        double n = stod(b_i.numerator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
        double d = stod(b_i.denominator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
        b_vals.push_back(make_pair(n, d));
        b_sum += n / d;
    }

    // normalize
    for (int i = 0; i < belief_size; ++i) {
        double val = b_vals[i].first / b_vals[i].second / b_sum;
        b_vals[i].first  = int(val * MyUtils::DOUBLE_ROUND_FACTOR);
        b_vals[i].second = MyUtils::DOUBLE_ROUND_FACTOR;
    }

    z3::expr b_sum_pre = model->eval(b_vars[belief_size + 2]);
    double b_sum_pre_val = stod(b_sum_pre.numerator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str()) /
            stod(b_sum_pre.denominator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());

    // tagged state
    z3::expr b_tagged = model->eval(b_vars[belief_size]);
    double n = stod(b_tagged.numerator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
    double d = stod(b_tagged.denominator().get_decimal_string(MyUtils::DOUBLE_DIGIS).c_str());
    double val = n / d / b_sum_pre_val;

    b_vals.push_back(make_pair(int(val * MyUtils::DOUBLE_ROUND_FACTOR), MyUtils::DOUBLE_ROUND_FACTOR));

    // robot location
    z3::expr b_rob_pos = model->eval(b_vars[belief_size + 1]);
    int rob_pos = b_rob_pos.get_numeral_int();
    b_vals.push_back(make_pair(rob_pos, 1));

    b_vals.push_back(make_pair(int(b_sum_pre_val * MyUtils::DOUBLE_ROUND_FACTOR), MyUtils::DOUBLE_ROUND_FACTOR));

    return make_shared <BeliefState> (b_vals);;

}

string TagDomain::getActionMeaning(int action_id) {
    string action_meaning = "";
    if (action_id == north_idx) {
        action_meaning = "move north";
    } else if (action_id == south_idx) {
        action_meaning = "move south";
    } else if (action_id == east_idx) {
        action_meaning = "move east";
    } else if (action_id == west_idx) {
        action_meaning = "move west";
    } else if (action_id == tag_idx) {
        action_meaning = "tag";
    } else if (action_id == stay_idx) {
        action_meaning = "stay";
    }
    return action_meaning;
}

string TagDomain::getObservationMeaning(int observation_id) {
    string observation_meaning = "";
    if (observation_id == yes_obs) {
        observation_meaning = "find target";
    } else if (observation_id == no_obs) {
        observation_meaning = "no target";
    } else if (observation_id == tag_success_obs) {
        observation_meaning = "success";
    }
    return observation_meaning;
}

int TagDomain::observe(BeliefStatePtr b, int action_id, int step) {

    auto b_vals = b->getBeliefStateValues();
    int curr_rob_loc = b_vals[belief_size + 1];
    if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
        cout << "previous belief : " << endl;
        cout << beliefStateToString(b, step) << endl;
        cout << "current target loc: " << curr_target_loc << endl;
    }

    if (action_id == north_idx || action_id == south_idx
        || action_id == west_idx || action_id == east_idx || action_id == stay_idx) {
        int next_rob_loc = getNextLoc(action_id, curr_rob_loc);
        vector<double> next_target_b = getNextTargetLocDist(curr_rob_loc, curr_target_loc);

        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "next target loc dist: ";
            for (int j = 0; j < belief_size; ++j) {
                if (next_target_b[j] > 0) {
                    cout << j << ": " << next_target_b[j] << ", ";
                }
            }
            cout << endl;
        }

        double rand_point = MyUtils::uniform01();
        int next_target_loc = curr_target_loc;
        for (int j = 0; j < belief_size; ++j) {
            if (rand_point < next_target_b[j]) {
                next_target_loc = j;
                break;
            }
            rand_point -= next_target_b[j];
        }
        if (MyUtils::debugLevel >= MyUtils::LEVEL_ONE_MSG) {
            cout << "new target loc : " << curr_target_loc << endl;
        }
        if (next_rob_loc == next_target_loc) {
            return yes_obs;
        }
        curr_target_loc = next_target_loc;
        return no_obs;
    } else if (action_id == tag_idx) {
        return tag_success_obs;
    } else {
        MyUtils::printErrorMsg("unknown action id: " + to_string(action_id));
    }

    return tag_success_obs;
}

void TagDomain::generateRandomInstances(string test_file_dir, int num) {
    ofstream test_file;
    test_file.open(test_file_dir + "/tag_test_" + to_string(1));
    test_file << tag_th << endl;
    test_file << rob_init_loc << endl;
    test_file << curr_target_loc << endl;
    test_file << epsilon << endl;
    test_file.close();

    for (int i = 2; i <= num; ++i) {
        ofstream test_file;
        test_file.open(test_file_dir + "/tag_test_" + to_string(i));
        test_file << tag_th << endl;
        int r_init = MyUtils::uniformInt(0, belief_size - 1);
        test_file << r_init << endl;
        int t_init = r_init;
        while (t_init == r_init) {
            t_init = MyUtils::uniformInt(0, belief_size - 1);
        }
        test_file << t_init << endl;
        test_file << epsilon << endl;
        test_file.close();
    }
}

int TagDomain::getStateIdx(int r, int i) {
    return r * (belief_size + 1) + i;
}

void TagDomain::computeAlphaVector(PolicyNodePtr policy) {
    unordered_map<int, double> alpha;
    int action_id = policy->getAction();

    auto child_nodes = policy->getChildNodes();

    if (MyUtils::debugLevel >= MyUtils::LEVEL_TWO_MSG) {
        if (action_id != 0) {
            cout << "alpha vector for action " << getActionMeaning(action_id) << ":" << endl;
        } else {
            cout << "terminal alpha vector" << endl;
        }
    }
    BeliefStatePtr b = policy->getBelief();
    auto b_vals = b->getBeliefStateValues();
    int curr_rob_loc = b_vals[belief_size + 1];

    for (int i = 0; i < belief_size; ++i) {
        double alpha_val = 0.0;
        if (!policy->isGoal()) {
            // non terminal node
            if (action_id == north_idx || action_id == south_idx
                || action_id == west_idx || action_id == east_idx || action_id == stay_idx) {
                // -1 for each motion action
                alpha_val += -1;

                int next_rob_loc = getNextLoc(action_id, curr_rob_loc);
                vector<double> next_target_loc_dist = getNextTargetLocDist(curr_rob_loc, i);

                if (child_nodes.find(yes_obs) != child_nodes.end()) {
                    PolicyNodePtr yes_child = child_nodes[yes_obs];
                    // deal with yes - only for j == next_rob_loc, Z(s', a, o) > 0
                    int j = next_rob_loc;
                    alpha_val += gamma * next_target_loc_dist[j]
                                 * yes_child->getAlpha(j);
                }

                if (child_nodes.find(no_obs) != child_nodes.end()) {
                    PolicyNodePtr no_child = child_nodes[no_obs];
                    for (int j = 0; j < belief_size; ++j) {
                        // deal with no - only for j == next_rob_loc, Z(s', a, o) == 0
                        if (j == next_rob_loc) {
                            continue;
                        }
                        alpha_val += gamma * next_target_loc_dist[j] * no_child->getAlpha(j);
                    }
                }

            } else if (action_id == tag_idx) {
                if (curr_rob_loc == i) {
                    alpha_val += 10;
                } else {
                    alpha_val -= 10;
                }
                PolicyNodePtr child = child_nodes[tag_success_obs];
                alpha_val += gamma * child->getAlpha(i);
            }
        }
        if (alpha_val != 0.0) {
            alpha[i] = alpha_val;
            if (MyUtils::debugLevel >= MyUtils::LEVEL_TWO_MSG) {
                cout << "for state: robot loc - " << curr_rob_loc << ", target loc - " << i
                     << ", alpha value: " << alpha[i] << endl;
            }
        }
    }

    policy->setAlpha(alpha);
};

double TagDomain::computeBeliefValue(BeliefStatePtr b,
                          unordered_map<int, double> &alpha, int step) {
    vector<double> b_vals = b->getBeliefStateValues();

    double value = 0.0;

    for (int i = 0; i < belief_size; ++i) {
        if (alpha.find(i) != alpha.end()) {
            value += b_vals[i] * alpha[i];
        }
    }
    return value;
}


unordered_map<int, bool> TagDomain::getAvailableActions() {
    unordered_map<int, bool> actions;
    actions[north_idx] = true;
    actions[south_idx] = true;
    actions[west_idx] = true;
    actions[east_idx] = true;
    actions[tag_idx] = true;
    // actions[stay_idx] = true;
    return actions;
};

BeliefStatePtr TagDomain::getNextBeliefDirect(BeliefStatePtr b_pre, int action_id,
                             int obs_id, int step) {
    if (isGoal(b_pre, step - 1)) {
        return nullptr;
    }
    vector<double> b_vals = b_pre->getBeliefStateValues();
    int curr_rob_loc = b_vals[belief_size + 1];
    if ((action_id == north_idx || action_id == south_idx
        || action_id == west_idx || action_id == east_idx || action_id == stay_idx)) {
        if (b_vals[curr_rob_loc] == 1) {
            // in the same cell
            return nullptr;
        }
        int next_rob_loc = getNextLoc(action_id, curr_rob_loc);
        if (next_rob_loc == curr_rob_loc) {
            return nullptr;
        }
        double yes_prob = 0.0;

        double b_sum_pre = 0.0;
        for (int i = 0; i < belief_size; ++i) {
            vector<double> next_target_loc_dist = getNextTargetLocDist(curr_rob_loc, i);
            yes_prob += next_target_loc_dist[next_rob_loc] * b_vals[i];
            b_sum_pre += b_vals[i];
        }
        yes_prob = yes_prob / b_sum_pre;
        if (obs_id == yes_obs) {
            if (yes_prob > yes_th) {
                for (int i = 0; i < belief_size; ++i) {
                    if (i == next_rob_loc) {
                        b_vals[i] = 1;
                    } else {
                        b_vals[i] = 0;
                    }
                }
                b_vals[belief_size] = 0;
                b_vals[belief_size + 1] = next_rob_loc;
                b_vals[belief_size + 2] = b_sum_pre;
                return make_shared <BeliefState> (b_vals);
            } else {
                return nullptr;
            }
        } else if (obs_id == no_obs) {
            if (yes_prob < 1 - yes_th) {
                vector<double> b_next_vals;
                for (int i = 0; i < belief_size; ++i) {
                    b_next_vals.push_back(0);
                }
                b_next_vals.push_back(yes_prob);
                b_next_vals.push_back(next_rob_loc);
                b_next_vals.push_back(b_sum_pre);

                double b_sum_pre = 0.0;
                for (int i = 0; i < belief_size; ++i) {
                    vector<double> next_target_loc_dist = getNextTargetLocDist(curr_rob_loc, i);
                    for (int j = 0; j < next_target_loc_dist.size(); ++j) {
                        if (j == next_rob_loc) {
                            continue;
                        }
                        b_next_vals[j] += next_target_loc_dist[j] * b_vals[i];
                    }
                    b_sum_pre += b_vals[i];
                }
                double b_sum = 0.0;
                for (int i = 0; i < belief_size; ++i) {
                    if (b_next_vals[i] / b_sum_pre < zero_th) {
                        b_next_vals[i] = 0;
                    }
                    b_sum += b_next_vals[i];

                }
                for (int i = 0; i < belief_size; ++i) {
                    b_next_vals[i] = b_next_vals[i] / b_sum;
                }
                return make_shared <BeliefState> (b_next_vals);
            } else {
                return nullptr;
            }

        }

    } else if (action_id == tag_idx && obs_id == tag_success_obs) {
        if (b_vals[curr_rob_loc] == 1) {
            double b_sum_pre = 0.0;
            for (int i = 0; i < belief_size; ++i) {
                vector<double> next_target_loc_dist = getNextTargetLocDist(curr_rob_loc, i);
                b_sum_pre += b_vals[i];
            }
            b_vals[belief_size] = b_vals[curr_rob_loc];
            b_vals[belief_size + 2] = b_sum_pre;
            return make_shared <BeliefState> (b_vals);

        } else {
            return nullptr;
        }
    }
    return nullptr;
}

bool TagDomain::isGoal(BeliefStatePtr b, int step) {
    auto b_vals = b->getBeliefStateValues();
    double tagged = b_vals[belief_size];
    return tagged > tag_th;
}

