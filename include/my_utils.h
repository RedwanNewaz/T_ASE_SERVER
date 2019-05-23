#ifndef MY_UTILS_H
#define MY_UTILS_H

#include <sstream>
#include <chrono>
#include <z3++.h>
#include <random>
#include <memory>
#include <unordered_map>

using namespace std;

// define smart pointers for z3 objects
typedef shared_ptr <z3::solver> SolverPtr;
typedef shared_ptr <z3::model> ModelPtr;
typedef shared_ptr <z3::context> ContextPtr;

/** time point */
typedef chrono::system_clock::time_point point;

/** time duration */
typedef chrono::system_clock::duration duration;

class BeliefState;
typedef shared_ptr <BeliefState> BeliefStatePtr;



namespace Color {
	enum Code {
		FG_RED      = 31,
		FG_GREEN    = 32,
		FG_BLUE     = 34,
		FG_DEFAULT  = 39,
		BG_RED      = 41,
		BG_GREEN    = 42,
		BG_BLUE     = 44,
		BG_DEFAULT  = 49
	};
	class Modifier {
		Code code;
	public:
		Modifier(Code pCode) : code(pCode) {}
		friend std::ostream&
		operator<<(std::ostream& os, const Modifier& mod) {
			return os << "\033[" << mod.code << "m";
		}
	};
}

static Color::Modifier blue(Color::FG_BLUE); // define in my_utils
static Color::Modifier green(Color::FG_GREEN);


class MyUtils {
public:
	static void printDebugMsg (string msg, int debug_level);
	static void printErrorMsg (string msg);
	static void setDebugLevel (int debug_level);
	static void printMsgToString(string msg, stringstream &ss);
    static void printZ3Expr (string msg, const z3::expr &exp, int debug_level);
	static point now();
	/** return the number of seconds that a time duration represents */
	static double run_time(const duration &d);
    static double uniform01 ();
    static double uniformReal(double lower_bound, double upper_bound);
    static int uniformInt(int lower_bound, int upper_bound);
    static int sampleObservation(const vector<pair<int, double>> &observation_dist);
    static double getAlphaForState(unordered_map<int, double> alpha, int idx);

    static bool checkBeliefEquality(BeliefStatePtr b1, BeliefStatePtr b2);

    const static int LEVEL_ZERO_MSG = 0;
	const static int LEVEL_ONE_MSG = 1;
	const static int LEVEL_TWO_MSG = 2;
	const static int DEBUG_BUFFER_SIZE = 10000;
    const static int DOUBLE_DIGIS = 6;
    const static int DOUBLE_ROUND_FACTOR;
    const static double DOUBLE_TOLENRANCE;
	const static string STAY_IN_GOAL_ACTION;
	const static string OBSERVATION_VAR_PREFIX;
	const static string ACTION_VAR_PREFIX;
	const static string BELIEF_VAR_PREFIX;
	const static string ROB_POS_VAR;
    static string solver_mode;
    static string solver_name;
    static int debugLevel;
    static mt19937 generator;
    static uniform_real_distribution<> uniDist;
};



#endif
