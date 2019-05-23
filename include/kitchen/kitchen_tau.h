//
// Created by Redwan Newaz on 2019-04-27.
//

#ifndef T_ASE_KITCHEN_TAU_H
#define T_ASE_KITCHEN_TAU_H

#include "belief_state.h"
#include "kitchen/kitchen_main.h"
#include <memory>
#include <unordered_map>
#include <cassert>
#include <algorithm>
/**@brief
 *  This class deals with transition and observation distribution \par
 *  This is the subclass of kitchen_main. \par
 *  However, transition distribution is ignored in online partial policy synthesis. \par
 *  kitchen_tau is the friend class of kitchne_main, so we can use all the function available for kitchen_main \par
 */

using namespace std;

class kitchen_tau {

public:
    kitchen_tau(kitchen_main *model, BeliefStatePtr b_pre, int action_id,
                int obs_id, int step);

    virtual ~kitchen_tau(); // deleting model_ will create error

    BeliefStatePtr compute();  // dependency - isInsideGrid, getMoveEffect, getLookEffect, getPickEffect, action_type

    vector<pair<int, double>> pdfZ();// dependency - action_type, getz


private:
    vector<double> b_vals;
    unordered_map<int, int> b_to_g_map_pre, b_to_g_map_next;
    kitchen_main * model_;
    int action_, obs_,step_;
    /** @param ACT : abstract definition of action \par
     * HACK: positive_observation = 2, negative_observation = 3 \par
     * HACK: pickup_positive_observation = 4 and pickup_negative_observation =5 \par
     * change act enum accordingly will help to create observation distribution \refitem pdfZ()
     */
    enum ACT{
        MOVE=1,
        LOOK=2,
        PICK=4
    };


protected:
    bool isInsideGrid(double b_sum_pre);
    bool getMoveEffect(vector<double>& b_next_vals, double b_sum_pre);
    bool getLookEffect(vector<double>& b_next_vals, double b_sum_pre);
    bool getPickEffect(vector<double>& b_next_vals, double b_sum_pre);
    ACT action_type(int action_id);
    tuple<double,double> getz(double fp, double fn, ACT type);




};


#endif //T_ASE_KITCHEN_TAU_H
