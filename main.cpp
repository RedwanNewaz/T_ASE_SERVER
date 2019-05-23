#include <benchmark/benchmark.h>
#include <iostream>
#include <thread>
#include <cassert>
#include "my_utils.h"
#include "solvers/solver.h"
#include "solvers/partial_solver.h"
#include <fstream>
#include <string>
//#include "tag/tag_main.h"
#include "kitchen/kitchen_main.h"
#include "pomdps/corridor_domain.h"


double stay_threshold;
int max_k ;
int debug_level=0;
string filename;
double PolicySynthesis(const char *testFile){
    MyUtils::solver_name = "partial";
    MyUtils::setDebugLevel(debug_level);
    point start = MyUtils::now();
    POMDPDomainPtr domain = make_shared<kitchen_main>(testFile);
    PolicySolverPtr solver = make_shared<PartialSolver>(stay_threshold,true);
    PolicyNodePtr policy = solver->solve(domain, 0, max_k);
    point end = MyUtils::now();
    double synthesis_time = MyUtils::run_time(end - start);
    return synthesis_time;
}


double old_policy_synthesis(const char * testFile)
{

    MyUtils::solver_name = "partial";
    MyUtils::setDebugLevel(debug_level);
    point start = MyUtils::now();
    POMDPDomainPtr domain = make_shared<CorridorDomain>(testFile);
    PolicySolverPtr solver = make_shared<PartialSolver>(stay_threshold);
    PolicyNodePtr policy = solver->solve(domain, 0, max_k);
    point end = MyUtils::now();
    double synthesis_time = MyUtils::run_time(end - start);
    return synthesis_time;
}


void updateParameter(benchmark::State& st)
{
    st.PauseTiming();
    // read log
    ifstream file("benchmark.log");
    if(file.is_open())
    {
        file>>stay_threshold;
        file>>max_k;
        file>>filename;
    }
    else
    {
        exit(-1);
    }
    file.close();

//    debug(stay_threshold<<"\t"<<max_k<<"\t"<<filename);
    st.ResumeTiming();
}



template<typename T>
class MyFixture : public benchmark::Fixture {};

BENCHMARK_TEMPLATE_F(MyFixture, DpTest, double)(benchmark::State& st) {
    updateParameter(st);
    for (auto _ : st) {
        double elapsed = PolicySynthesis(filename.c_str());
//        benchmark::DoNotOptimize(filename);
        st.counters["elapsed"] = elapsed;

    }
}

BENCHMARK_TEMPLATE_F(MyFixture, RecursionTest, double)(benchmark::State& st) {
    updateParameter(st);
    for (auto _ : st) {
        double elapsed = old_policy_synthesis(filename.c_str());
//        benchmark::DoNotOptimize(filename);
        st.counters["elapsed"] = elapsed;

    }
}


BENCHMARK_REGISTER_F(MyFixture, RecursionTest);

BENCHMARK_REGISTER_F(MyFixture, DpTest);

BENCHMARK_MAIN();
