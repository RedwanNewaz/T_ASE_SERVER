//
// Created by Redwan Newaz on 2019-04-27.
//

#include "kitchen/kitchen_tau.h"


kitchen_tau::kitchen_tau(kitchen_main *model, BeliefStatePtr b_pre, int action_id, int obs_id, int step) :
        model_(model), action_(action_id), obs_(obs_id), step_(step){

    b_vals = b_pre->getBeliefStateValues();
    b_to_g_map_pre = model_->belief_to_group_maps[step-1];
    b_to_g_map_next = model_->belief_to_group_maps[step];

}

BeliefStatePtr kitchen_tau::compute() {
    double b_sum_pre = 0;
    for (int g = 0; g < b_to_g_map_pre.size(); ++g) {
        b_sum_pre += b_vals[g] * b_to_g_map_pre[g];
    }


    vector<double> b_next_vals;
    bool update = false;
    switch (action_type(action_)){
        case MOVE: update=getMoveEffect(b_next_vals, b_sum_pre);break;
        case LOOK: update=getLookEffect(b_next_vals,b_sum_pre);break;
        case PICK: update=getPickEffect(b_next_vals, b_sum_pre); break;
    }

    return update?make_shared <BeliefState> (b_next_vals): nullptr;
}

bool kitchen_tau::isInsideGrid(double b_sum_pre ) {
    int action_id = action_;
    double r_top = 0;
    double r_bottom = 0;
    double r_left = 0;
    double r_right = 0;

    // TODO sanity check if the robot goes outside of the gird
    for (auto b_to_g : b_to_g_map_pre) {
        int idx = b_to_g.first;
        int r = model_->getRobotLocIdx(idx);
        int g_idx_pre = b_to_g.second;

        double b_g_pre = b_vals[g_idx_pre];
        if (model_->getNextLoc(move_west_idx, r) == r) {
            r_left += b_g_pre;
        }
        if (model_->getNextLoc(move_east_idx, r) == r) {
            r_right += b_g_pre;
        }
        if (model_->getNextLoc(move_south_idx, r) == r) {
            r_bottom += b_g_pre;
        }
        if (model_->getNextLoc(move_north_idx, r) == r) {
            r_top += b_g_pre;
        }
    }

    if ((action_id == move_north_idx && r_top > model_->reach_th * b_sum_pre)
        || (action_id == move_south_idx && r_bottom > model_->reach_th * b_sum_pre)
        || (action_id == move_west_idx && r_left > model_->reach_th * b_sum_pre)
        || (action_id == move_east_idx && r_right > model_->reach_th * b_sum_pre)) {
        return false;
    }
    return true;
}

bool kitchen_tau::getMoveEffect(vector<double>& b_next_vals, double b_sum_pre) {

    if(!isInsideGrid(b_sum_pre))
        return false;
    size_t B_G_N = b_to_g_map_next.size(); // belief group number
    vector<double> move_effects(B_G_N, 0.0);
    vector<bool> has_pre(B_G_N, false);
    vector<bool> has_next(B_G_N, false);

    for (auto b_to_g : b_to_g_map_pre) {
        int idx_pre = b_to_g.first;
        int r = model_->getRobotLocIdx(idx_pre);
        int r_next = model_->getNextLoc(action_, r);
        int i = model_->getObsLocsIdx(idx_pre);
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
            int idx_next = model_->getStateIdx(r_next, i);
            if (!has_next[b_to_g_map_next[idx_next]]) {
                move_effects[b_to_g_map_next[idx_next]] += (b_g_pre * model_->p_succ);
                has_next[b_to_g_map_next[idx_next]] = true;
            }
            // fail effect
            if (!has_pre[b_to_g_map_next[idx_pre]]) {
                move_effects[b_to_g_map_next[idx_pre]] += (b_g_pre * (1 - model_->p_succ));
                has_pre[b_to_g_map_next[idx_pre]] = true;
            }
        }
    }
// STEP update value
    b_next_vals.resize(B_G_N);
    for (int g = 0; g < B_G_N; ++g) {
        b_next_vals[g] = move_effects[g];
    }
    // check safety
    double unsafe_prob = 0;
    double b_sum = 0;

    for (auto b_to_g : b_to_g_map_next) {
        int idx = b_to_g.first;
        int r = model_->getRobotLocIdx(idx);
        int i = model_->getObsLocsIdx(idx);
        int g_idx = b_to_g.second;

        if (model_->in_collision(r, model_->obs_locs_list[i])) {
            unsafe_prob += b_next_vals[g_idx];
        }
        b_sum += b_next_vals[g_idx];

    }

    if (unsafe_prob < model_->delta * b_sum) {
        for (int g = 0; g < B_G_N; ++g) {
            b_next_vals[g] = b_next_vals[g] / b_sum;
        }
    return true;
    }
    return false;
}

bool kitchen_tau::getLookEffect(vector<double>& b_next_vals, double b_sum_pre) {
    double potential_unsafe_prob = 0;
    double p_fp = model_->p_fp;
    double p_fn = model_->p_fn;


    for (auto b_to_g : b_to_g_map_pre) {
        int idx = b_to_g.first;
        int r = model_->getRobotLocIdx(idx);
        int look_loc =model_->getNextLoc(action_, r);
        if (look_loc == r) {
            // no effect for look
            continue;
        }
        int i = model_->getObsLocsIdx(idx);
        int g_idx_pre = b_to_g.second;

        if (model_->in_collision(look_loc, model_->obs_locs_list[i])) {
            potential_unsafe_prob += b_vals[g_idx_pre];
        }

    }

    if (potential_unsafe_prob > model_->delta * b_sum_pre && potential_unsafe_prob < 0.6 * b_sum_pre) {
        vector<double> look_pos_effects;
        vector<double> look_neg_effects;
        for (int g = 0; g < b_to_g_map_next.size(); ++g) {
            look_pos_effects.push_back(0);
            look_neg_effects.push_back(0);
        }

        for (auto b_to_g : b_to_g_map_pre) {
            int idx_pre = b_to_g.first;
            int r = model_->getRobotLocIdx(idx_pre);
            int look_loc = model_->getNextLoc(action_, r);
            int i = model_->getObsLocsIdx(idx_pre);
            int g_idx_pre = b_to_g.second;

            int g_idx_next = b_to_g_map_next[idx_pre];

            double b_g_pre = b_vals[g_idx_pre];
            if (look_loc == r) {
                look_pos_effects[g_idx_next] = (b_g_pre * p_fp);
                look_neg_effects[g_idx_next] = (b_g_pre * (1 - p_fp));
            } else {
                if (model_->in_collision(look_loc, model_->obs_locs_list[i])) {
                    look_pos_effects[g_idx_next] = (b_g_pre * (1 - p_fn));
                    look_neg_effects[g_idx_next] = (b_g_pre * p_fn);
                } else {
                    look_pos_effects[g_idx_next] = (b_g_pre * p_fp);
                    look_neg_effects[g_idx_next] = (b_g_pre * (1 - p_fp));

                }
            }
        }
// STEP update value
        double b_sum = 0;
        if (obs_ == positive_observation) {
            for (int g = 0; g < b_to_g_map_next.size(); ++g) {
                b_next_vals.push_back(look_pos_effects[g]);
                b_sum += b_next_vals[g] * b_to_g_map_next[g];
            }
        } else if (obs_ == negative_observation) {
            for (int g = 0; g < b_to_g_map_next.size(); ++g) {
                b_next_vals.push_back(look_neg_effects[g]);
                b_sum += b_next_vals[g] * b_to_g_map_next[g];
            }
        }
        for (int g = 0; g < b_to_g_map_next.size(); ++g) {
            b_next_vals[g] = b_next_vals[g] / b_sum;
        }
        return true;
    }
    return false;
}

bool kitchen_tau::getPickEffect(vector<double> &b_next_vals, double b_sum_pre) {
    double ready_prob = 0;
    for (int r = 0; r < model_->rob_locs_size; ++r) {
        auto xy = model_->getXY(r);
        int x = xy.first;
        if (x != model_->x_size - 1) {
            continue;
        }
        for (int i = 0; i < model_->obs_locs_list.size(); ++i) {
            int idx = model_->getStateIdx(r, i);
            if (b_to_g_map_pre.find(idx) == b_to_g_map_pre.end()) {
                continue;
            }
            int g_idx = b_to_g_map_pre[idx];
            ready_prob += b_vals[g_idx];
        }
    }
    if (ready_prob > model_->reach_th * b_sum_pre) {
        double r_pos_prob;
        double r_neg_prob;
        double u_pos_prob;
        double u_neg_prob;
        double g_pos_prob;
        double g_neg_prob;
        if (action_ == pick_up_left_hand_idx) {
            r_pos_prob = 0.0;
            r_neg_prob = 0.0;
            u_pos_prob = 0.03;
            u_neg_prob = 0.07;
            g_pos_prob = 0.72;
            g_neg_prob = 0.18;
        } else if (action_ == pick_up_right_hand_idx) {
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
        for (int g = 0; g < b_to_g_map_pre.size(); ++g) {
            g_pre_used.push_back(false);
        }
        vector<double> pickup_pos_effects;
        vector<double> pickup_neg_effects;

        for (int g = 0; g < b_to_g_map_next.size(); ++g) {
            pickup_pos_effects.push_back(0);
            pickup_neg_effects.push_back(0);
        }

        for (auto b_to_g : b_to_g_map_pre) {
            int idx_pre = b_to_g.first;
            int r = model_->getRobotLocIdx(idx_pre);
            int i = model_->getObsLocsIdx(idx_pre);
            int g_idx_pre = b_to_g.second;

            auto xy = model_->getXY(r);
            int x = xy.first;

            if (g_pre_used[g_idx_pre]) {
                continue;
            }
            g_pre_used[g_idx_pre] = true;
            double b_g_pre = b_vals[g_idx_pre] * b_to_g_map_pre[g_idx_pre];
            if (x == model_->x_size - 1) {
                int ready_idx = model_->getStateIdx(model_->ready_robot_loc, i);
                int g_ready_idx = b_to_g_map_next[ready_idx];
                pickup_pos_effects[g_ready_idx] += (b_g_pre * r_pos_prob);
                pickup_neg_effects[g_ready_idx] += (b_g_pre * r_neg_prob);

                int unsafe_idx = model_->getStateIdx(model_->unsafe_robot_loc, i);
                int g_unsafe_idx = b_to_g_map_next[unsafe_idx];
                pickup_pos_effects[g_unsafe_idx] += (b_g_pre * u_pos_prob);
                pickup_neg_effects[g_unsafe_idx] += (b_g_pre * u_neg_prob);

                int goal_idx = model_->getStateIdx(model_->goal_robot_loc, i);
                int g_goal_idx = b_to_g_map_next[goal_idx];
                pickup_pos_effects[g_goal_idx] += (b_g_pre * g_pos_prob);
                pickup_neg_effects[g_goal_idx] += (b_g_pre * g_neg_prob);
            } else {
                int g_idx_next = b_to_g_map_next[idx_pre];
                pickup_pos_effects[g_idx_next] += (b_g_pre * other_pos_prob);
                pickup_neg_effects[g_idx_next] += (b_g_pre * other_neg_prob);
            }
        }

// STEP update value
        double b_sum = 0;
        if (obs_ == pickup_positive_observation) {
            for (int g = 0; g < b_to_g_map_next.size(); ++g) {
                b_next_vals.push_back(pickup_pos_effects[g] / b_to_g_map_next[g]);
                b_sum += pickup_pos_effects[g];
            }
        } else {
            for (int g = 0; g < b_to_g_map_next.size(); ++g) {
                b_next_vals.push_back(pickup_neg_effects[g] / b_to_g_map_next[g]);
                b_sum += pickup_neg_effects[g];
            }
        }
        double goal_prob = 0;
        double unsafe_prob = 0;
        double ready_prob = 0;
        for (int i = 0; i < model_->obs_locs_list.size(); ++i) {
            int goal_idx = model_->getStateIdx(model_->goal_robot_loc, i);
            if (b_to_g_map_next.find(goal_idx) != b_to_g_map_next.end()) {
                int g_goal_idx = b_to_g_map_next[goal_idx];
                goal_prob += b_next_vals[g_goal_idx];
            }
            int unsafe_idx = model_->getStateIdx(model_->unsafe_robot_loc, i);
            if (b_to_g_map_next.find(unsafe_idx) != b_to_g_map_next.end()) {
                int g_unsafe_idx = b_to_g_map_next[unsafe_idx];
                unsafe_prob += b_next_vals[g_unsafe_idx];
            }
            int ready_idx = model_->getStateIdx(model_->ready_robot_loc, i);
            if (b_to_g_map_next.find(ready_idx) != b_to_g_map_next.end()) {
                int g_ready_idx = b_to_g_map_next[ready_idx];
                ready_prob += b_next_vals[g_ready_idx];
            }
        }
        if (unsafe_prob < 0.2 * (goal_prob + unsafe_prob + ready_prob)) {
            for (int g = 0; g < b_to_g_map_next.size(); ++g) {
                b_next_vals[g] = b_next_vals[g] / b_sum;
            }
            return true;
        }
    }
    return false;
}

kitchen_tau::~kitchen_tau() {

}


kitchen_tau::ACT kitchen_tau::action_type(int action_id) {
    if (action_id == move_north_idx || action_id == move_south_idx
        || action_id == move_west_idx || action_id == move_east_idx)
        return MOVE;
    else if (action_id == look_north_idx || action_id == look_south_idx
             || action_id == look_west_idx || action_id == look_east_idx)
        return LOOK;
    else if (action_id == pick_up_left_hand_idx || action_id == pick_up_right_hand_idx)
        return PICK;
}

tuple<double, double> kitchen_tau::getz(double fp, double fn, kitchen_tau::ACT type) {
    double p_pos = 0.0;
    double p_neg = 0.0;

    double other_pos_prob ;
    double other_neg_prob ;

    if (action_ == pick_up_left_hand_idx){
        other_pos_prob= 0.75;
        other_neg_prob =0.25;
    }
    else{
        other_pos_prob= 0.8;
        other_neg_prob =0.2;
    }

    for (int r = 0; r < model_->rob_locs_size; ++r){
        for (int i = 0; i < model_->obs_locs_list.size(); ++i) {
            int idx_pre = model_->getStateIdx(r, i);
            if (b_to_g_map_pre.find(idx_pre) == b_to_g_map_pre.end()) {
                continue;
            }
            int s = b_to_g_map_pre[idx_pre];
            double b = b_vals[s];
            if (type == LOOK) {
                int look_loc = model_->getNextLoc(action_, r);
                if (model_->in_collision(look_loc, model_->obs_locs_list[i])){
                    p_pos += b * (1-fn);
                    p_neg += b * fn;
                }
                else{
                    p_pos += b * fp;
                    p_neg += b * (1 - fp);
                }
            } else if(type ==PICK) {
                p_pos += b * other_pos_prob;
                p_neg += b * other_neg_prob;
            }


        }
    }
    return make_tuple(p_pos,p_neg);
}

vector<pair<int, double>> kitchen_tau::pdfZ() {
    vector<pair<int, double>> observations;
    double p_pos, p_neg;
    ACT u = action_type(action_);
    switch (u){
        case MOVE:
        {
            observations.push_back(make_pair (trans_observation, 1.0));
            return observations;
        }
        case LOOK: tie(p_pos, p_neg)=getz(model_->p_fp, model_->p_fn, LOOK); break;
        case PICK: tie(p_pos, p_neg)=getz(model_->p_fp, model_->p_fn, PICK); break;
        default:
            assert("unknown action");
    }

    obs_ = static_cast<int>(u);
    observations.push_back(make_pair(obs_, p_pos / (p_pos + p_neg)));
    observations.push_back(make_pair(++obs_, p_neg / (p_pos + p_neg)));

    return observations;
}
