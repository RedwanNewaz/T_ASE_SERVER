#ifndef POLICY_SYNTHESIER_CC
#define POLICY_SYNTHESIER_CC

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <getopt.h>

#include "my_utils.h"
#include "pomdps/corridor_domain.h"
#include "pomdps/lab_domain.h"
#include "pomdps/tag_domain.h"
#include "solvers/solver.h"
#include "solvers/partial_solver.h"
#include "solvers/quantitative_solver.h"

using namespace std;
double stay_threshold = 1.0;
int max_k = 20;
int debug_level = 1;
int test_num = 0;
string domain_name = "corridor";
string file_name = "output_corridor_test_4_4.log";
string test_file_path = "/Users/redwannewaz/Projects/yue/boolean_pomdp_synthesizer/tests/ral_tests/corridor_test_4";
//string test_file_path = "/Users/redwannewaz/Projects/yue/boolean_pomdp_synthesizer/tests/ral_tests/test_instances/tag_test_0.6";
string test_instance_dir = "../../tests/ral_tests";

PolicySolverPtr runSolver() {
    POMDPDomainPtr domain = nullptr;
    point start = MyUtils::now();
    if (domain_name == "corridor") {
        domain = make_shared <CorridorDomain> (test_file_path);
    } else if (domain_name == "lab") {
        domain = make_shared <LabDomain> (test_file_path);
    } else if (domain_name == "tag") {
        domain = make_shared <TagDomain> (test_file_path);
    } else {
        cout << "Error: unknown domain name" << endl;
        return nullptr;
    }
    PolicySolverPtr solver = nullptr;
    if (MyUtils::solver_name == "partial") {
        cout<<"partial solver selected"<<endl;
        solver = make_shared <PartialSolver> (stay_threshold);
    } else  if (MyUtils::solver_name == "quantitative") {
        cout<<"quantitative solver selected"<<endl;
        solver = make_shared <QuantitativeSolver> ();
    } else {
        cout<<"no solver selected"<<endl;
        return nullptr;
    }

    PolicyNodePtr policy = solver->solve(domain, 0, max_k);
    point end = MyUtils::now();

    solver->synthesis_time = MyUtils::run_time(end - start);
    return solver;
}

void genTests() {
    POMDPDomainPtr domain = nullptr;
    point start = MyUtils::now();
    if (domain_name == "corridor") {
        domain = make_shared <CorridorDomain> (test_file_path);
    } else if (domain_name == "lab") {
        domain = make_shared <LabDomain> (test_file_path);
    } else if (domain_name == "tag") {
        domain = make_shared <TagDomain> (test_file_path);
    }
    if (domain != nullptr) {
        domain->generateRandomInstances(test_instance_dir, test_num);
    }
}


int main (int argc, char** argv) {

    stringstream ss;

    bool generate_tests = false;
    MyUtils::solver_name ="partial";
	/* Parse Options */
    /*
    {
        int c;

        while ((c = getopt (argc, argv, "d:s:m:f:i:t:p:k:b:v:o:g:?")) != -1) {
            switch(c) {
			case 'd':
				domain_name = optarg;
				break;
            case 'k':
                max_k = strtol (optarg, NULL, 10);
                break;
            case 'o':
                file_name = optarg;
                break;
            case 'm':
                MyUtils::solver_mode = optarg;
                break;
            case 's':
                MyUtils::solver_name = optarg;
                break;
            case 'f':
                test_file_path = optarg;
                break;
            case 'i':
                test_instance_dir = optarg;
                break;
            case 'v':
                debug_level = strtol (optarg, NULL, 10);
                break;
            case 'g':
                test_num = strtol (optarg, NULL, 10);
                break;
            case 't':
                stay_threshold = stod(string(optarg));
                break;
            case '?':
                puts("Usage: ./policy_synthesizer [OPTIONS]\n"
                     "Policy synthesizer for task and motin planning in uncertain domains"
                     "\n"
                     "Options:\n"
					 "  -d DOMAIN_NAME       problem domain name (default: corridor)\n"
                     "  -o FILE_NAME         solver name (default: output.log)\n"
                     "  -m MODE              specify the solver mode (default: \"\")\n"
                     "  -s SOLVER_NAME       specify the solver (default: \"partial\")\n"
                     "  -f TEST_FILE_PATH    specify test file path (default: \"\")\n"
                     "  -t STAY_PROB_TH      specify the stay probability threshold for partial solver (default: 1.0)\n"
                     "  -k NUM_STEPS         max number of steps for forwardsearch (default: 1)\n"
                     "  -g NUM_TESTS         generate a number of tests (default: 0)\n"
                     "  -i TEST_INSTANCE_DIR specify the directory for placing generated tests (default: \"\")\n"
                     "  -v DEBUG_LEVEL       debug output level (default: 0)\n"
                     "\n"
                     "\n"
					);
                exit (EXIT_SUCCESS);
                break;
            default:
				cout << "Unknown parameter: " << c << endl;
            }

        }
    }
     */
	// set debug level
	MyUtils::setDebugLevel(debug_level);
    char msgBuffer[MyUtils::DEBUG_BUFFER_SIZE];

    if (test_num > 0) {
        genTests();
        return 0;
    }

    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "Solver Name: %s", MyUtils::solver_name.c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);

    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "Solver Mode: %s", MyUtils::solver_mode.c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);

    snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
              "Partial Solver Stay Prob Threshold: %f", stay_threshold);
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);
	snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
			  "Problem Domain: %s", domain_name.c_str());
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);

	snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
			  "Max number of steps for forward search: %d", max_k);
    MyUtils::printMsgToString(msgBuffer, ss);
    MyUtils::printDebugMsg(msgBuffer,
                           MyUtils::LEVEL_ONE_MSG);


    auto solver = runSolver();
    double synthesis_time;
    if (solver != nullptr) {
        synthesis_time = solver->synthesis_time;
        ss << solver->exec_info << endl;
        if (solver->success) {
            snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                      "finished execution: success");
            MyUtils::printMsgToString(msgBuffer, ss);
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
        } else {
            snprintf(msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                     "finished execution: fail");
            MyUtils::printMsgToString(msgBuffer, ss);
            MyUtils::printDebugMsg(msgBuffer,
                                   MyUtils::LEVEL_ONE_MSG);
        }
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "synthesis time: %fs", synthesis_time);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "number of plans checked: %d", solver->num_plan_checked);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "planning number: %d", solver->one_run_planning_num);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "Z3 Solver checking time: %fs", solver->solver_checking_time);
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
        snprintf (msgBuffer, MyUtils::DEBUG_BUFFER_SIZE,
                  "average solver checking time per call: %fms",
                  1000 * (solver->solver_checking_time / solver->num_plan_checked));
        MyUtils::printMsgToString(msgBuffer, ss);
        MyUtils::printDebugMsg(msgBuffer,
                               MyUtils::LEVEL_ONE_MSG);
    }
    ofstream log_file;
    log_file.open (file_name);
    log_file << ss.str() << endl;
    log_file.close();
	return 0;
}

#endif

