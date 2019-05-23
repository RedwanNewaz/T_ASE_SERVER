//
// Created by Redwan Newaz on 2018-12-23.
//

#include "pomdps/navigation_domain.h"
#include <cassert>

navigation_domain::navigation_domain(int s, int a_s, int o_s) : POMDPDomain(s, a_s, o_s) {
    //TODO: initialize the POMDP domain with following parameters
    //STEP: initial belief
    vector<pair<double, double>> init_b_vals;
    init_b_vals.push_back(make_pair(1.0, m_grid->NumBlocks()));
    init_belief = make_shared <BeliefState> (init_b_vals);
    //STEP belief map
    belief_to_group_maps.clear();
    belief_group_size.clear();
    belief_group_size.resize(start_step+1);
    belief_to_group_maps.resize(start_step+1);
    belief_group_size[start_step].push_back(m_grid->NumBlocks());
    init_robot_loc = 0;
    for (int i = 0; i <m_grid->NumBlocks() ; ++i) {
        //TODO: getStateIndex given the robot location and update belief to group map
        int idx = init_robot_loc<<i;
        belief_to_group_maps[start_step][idx] = 0;
    }
    //STEP : extractProbs
    //STEP: FilterZeroProb
}

POMDPDomainPtr navigation_domain::createNewPOMDPDomain(int s, BeliefStatePtr init_b, int a_s, int o_s) {
   //STEP dependency - filterZeroProb and new constructor
//    return make_shared <navigation_domain> (s, filterZeroProb(init_b, s),
////                                         a_s, o_s, belief_to_group_maps[s]);
    return nullptr;
}

z3::expr navigation_domain::genTransCond(z3::context &ctx, int step) {

    assert((step<=belief_group_size.size() && " incorrect belief group map update"));



    // TODO there are two action categories - MOVE and LOOK, group the belief based on these categories
/* GARBAGE - don't need
    enum group{MOVE =0, LOOK };
    vector<vector<int>> ACTION{{move_east_idx,move_north_idx,move_west_idx,move_south_idx},
                      {look_east_idx,look_north_idx,look_west_idx,look_south_idx}};

    auto get_group = [ACTION](int type, int action_idx){
        auto act = ACTION[type];
        auto res = std::find(act.begin(),act.end(),action_idx);
        return (res != act.end());};
    auto is_move_group = std::bind(get_group,MOVE,std::placeholders::_1);
    auto is_look_group = std::bind(get_group,LOOK,std::placeholders::_1);
*/

    // STEP - make the second element of each b in b_to_g_map_pre as string (value), but why?
    auto &b_to_g_map_pre = belief_to_group_maps[step - 1];
    auto stateTocoord = [col=m_grid->col](int index){ vector<unsigned long> coord {index%col,index/col}; return coord;};
    for (auto b : b_to_g_map_pre) {
        int r = b.first>>m_grid->NumBlocks();
        // STEP - get the next index
        auto robot = stateTocoord(r);
        int  _loc(0);
        // STEP - update the state iff the position is valid - inside the grid and not on the obstacles
        for (float i = 0; i <2*M_PI ; i+=M_PI_2, ++_loc) {
            int next_state = b.first;
            int x = robot[0] + cos(i);
            int y = robot[1] + sin(i);
            if((y>=0 && y<m_grid->row) && (x>=0 && x <m_grid->col) && m_grid->Collision(x,y))
                next_state= m_grid->CoordToIndex(x,y)<<m_grid->NumBlocks();
            //TODO how to utilize/ save the next state ?
        }
    }



    z3::expr_vector trans_conds(ctx);
    z3::expr res(ctx);
    return res;
}

z3::expr navigation_domain::genGoalCond(z3::context &ctx, int step) {

    // STEP dependency - goal_robot_loc, unsafe_robot_loc, ready_robot_loc, getStateIdx
    auto &b_to_g_map = belief_to_group_maps[step];
    int size = b_to_g_map.size();
    int num_belief = 3; // goal_g, unsafe_g, ready_g
    vector<vector<int>> container(num_belief, vector<int> (size, 0));


    //STEP - compute the number of uncertain cells - last row is for goal locations
    int num_grid_cells = (m_grid->row)*m_grid->col; // TODO: replaced by rob_locs_size
    int num_uncertain_cells = (m_grid->row-1)*m_grid->col; // TODO: replaced by ready robot location
    int num_obs_states = nCr(num_uncertain_cells,m_grid->NumBlocks());

    // TODO: replace these values
    int ready_robot_loc = num_uncertain_cells;
    int unsafe_robot_loc = ready_robot_loc + 1;//
    int goal_robot_loc = unsafe_robot_loc + 1;

    //STEP update the individual trajectory
    auto getStateIdx = [](int s, int i){ return s<<i;};
    for (int i = 0; i <num_obs_states ; ++i) {
        int goal_idx = getStateIdx(goal_robot_loc, i);
        int unsafe_idx = getStateIdx(unsafe_robot_loc, i);
        int ready_idx = getStateIdx(ready_robot_loc, i);
        int index[] = {goal_idx,unsafe_idx,ready_idx};
        for (int j = 0; j < num_belief; ++j) {
            if (b_to_g_map.find(index[j]) != b_to_g_map.end()) {
                int _index = b_to_g_map[index[j]];
                ++container[j][_index];
            }
        }

    }


    //STEP declare z3 variables
    string prefix = MyUtils::BELIEF_VAR_PREFIX;
    auto var_name = [prefix,k=step](int i) { return prefix+"_"+to_string(i)+"_"+to_string(k);};
    z3::expr_vector goal_states(ctx);
    z3::expr_vector unsafe_states(ctx);
    unsafe_states.push_back(ctx.int_val(0));
    z3::expr_vector ready_states(ctx);
    ready_states.push_back(ctx.int_val(0));
    //STEP populate z3 variables
    for (int l = 0; l < size; ++l) {
        auto b_g_k_name = var_name(l);
        auto b_g_k = ctx.real_const(b_g_k_name.c_str());
        for (int i = 0; i <num_belief ; ++i) {
            bool update = (container[i][l] != 0);
            if(!update) continue;
            switch (i){
                case 0:goal_states.push_back  (b_g_k * container[i][l]);break;
                case 1:unsafe_states.push_back(b_g_k * container[i][l]);break;
                case 2:ready_states.push_back (b_g_k * container[i][l]);break;
            }
        }

    }

    //STEP update other variables - action and observation
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

    //STEP - we don't need the final action that is to pick up using any of hands.
    goal_conds.push_back((a_k == ctx.int_val(pick_up_left_hand_idx))
                         || (a_k == ctx.int_val(pick_up_right_hand_idx)));


    return  mk_and(goal_conds);
}

z3::expr navigation_domain::genInitCond(z3::context &ctx) {
    // STEP dependency - see_obs_num
    int see_obs_num =0;

    z3::expr_vector init_conds(ctx);
    vector<z3::expr> b_s_vals = init_belief->getBeliefStateExprs(ctx);

    //STEP generate string names for variables
    string prefix = MyUtils::BELIEF_VAR_PREFIX;
    auto var_name = [prefix,i=0,k=start_step]()mutable { return prefix+"_"+to_string(i++)+"_"+to_string(k);};
    const int size = belief_group_size[start_step].size();
    int num_var = size + 2;

    assert(size == b_s_vals.size());

    vector<string> variables(num_var);
    std::generate_n(variables.begin(),size,var_name);
    variables[size] = MyUtils::BELIEF_VAR_PREFIX + "_" + "obs_num_" + std::to_string(start_step);
    variables[size + 1] = MyUtils::ACTION_VAR_PREFIX + "_" + std::to_string(start_step);


    //STEP populate z3 variables

//    init_conds.resize(num_var);
    for (int i = 0; i < num_var; ++i) {
        if (i==size) {
            z3::expr obs_num_var = ctx.int_const(variables[i].c_str());
            init_conds.push_back(obs_num_var == ctx.int_val(see_obs_num));
        } else if(i==size+1){
            z3::expr a_s_var = ctx.int_const(variables[i].c_str());
            init_conds.push_back(a_s_var == ctx.int_val(a_start));
        }
        else {
            auto b_s_vars = ctx.real_const(variables[i].c_str());
            init_conds.push_back(b_s_vars == b_s_vals[i]);
        }

    }

    return mk_and(init_conds);
}

vector<z3::expr> navigation_domain::getBeliefStateVarList(z3::context &ctx, int step) {
    //TODO: we don't need this function - remove it
    return vector<z3::expr>();
}

BeliefStatePtr navigation_domain::getBeliefState(z3::context &ctx, ModelPtr model, int step) {
    // STEP dependency - belief_group_size
    // TODO: didn't change the original implementation
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

string navigation_domain::beliefStateToString(BeliefStatePtr b, int step) {
    //TODO: state visualization
    //STEP dependency - extractProbs(b,step)
    return std::string();
}

string navigation_domain::getActionMeaning(int action_id) {
    return std::string();
}

string navigation_domain::getObservationMeaning(int observation_id) {
    return std::string();
}

vector<pair<int, double>> navigation_domain::getObservationDistribution(int action_id, BeliefStatePtr b_pre, int step) {
    return vector<pair<int, double>>();
}

int navigation_domain::observe(BeliefStatePtr b, int action_id, int step) {
    //TODO: return observation based on the action
    //TODO: check the action leads to collision
    //STEP dependency: MyUtils::uniform01()
    //TODO: add noise to the observation
    // STEP - implement despot step function
    return 0;
}

void navigation_domain::generateRandomInstances(string test_file_dir, int num) {
    // TODO: don't need this function - remove it
}

void navigation_domain::computeAlphaVector(PolicyNodePtr policy) {
    // STEP dependency - rob_locs_size, obs_locs_list, avoid_regions
    // TODO - compute reward for the given action (reward is defined as action value)
}

double navigation_domain::computeBeliefValue(BeliefStatePtr b, unordered_map<int, double> &alpha, int step) {
    // STEP dependency - belief_to_group_maps
    // TODO: didn't change the original implementation
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

unordered_map<int, bool> navigation_domain::getAvailableActions() {
    // TODO return the NESW look and move actions
    return unordered_map<int, bool>();
}

BeliefStatePtr navigation_domain::getNextBeliefDirect(BeliefStatePtr b_pre, int action_id, int obs_id, int step) {
    // TODO the heart of the class - very long !!!
    return BeliefStatePtr();
}

bool navigation_domain::isGoal(BeliefStatePtr b, int step) {
    //TODO: robot location is given by init_robot_loc - check whether or not it is in the goal locations

    // TODO: check the probability along the trajectory satisfy the threshold constraints
    return false;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------MY FUNCTIONS------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------------------
navigation_domain::navigation_domain(std::shared_ptr<GridWorld> _grid):m_grid(_grid),POMDPDomain(0, 0, 0)
{
    debug("GridWorld "<< *m_grid);
    //STEP initialization
    vector<pair<double, double>> init_b_vals;
    //TODO: initialize belief_group_maps based on obstacle size

}

vector<vector<int>> navigation_domain::listCombination(int n, int r) {
    int num_states = nCr(n,r);
    vector<vector<int>> combList(num_states);
    std::vector<bool> v(n);
    std::fill(v.end() - r, v.end(), true);
    int count = 0;
    do {
        vector<int> rowVal;
        for (int i = 0; i < n; ++i)
            if (v[i])
                rowVal.emplace_back(i<<r); // TODO: need to change?
        combList[count]=rowVal;
        ++count;
    } while (std::next_permutation(v.begin(), v.end()));
    return combList;
}

int navigation_domain::nCr(int n, int k) {
    if (k > n) {
        return 0;
    }
    int r = 1;
    for (int d = 1; d <= k; ++d) {
        r *= n--;
        r /= d;
    }
    return r;
}

