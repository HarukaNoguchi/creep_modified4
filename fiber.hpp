#pragma once
#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h> 
#include <mutex>


struct Fiber{
    double z=0;
    double f;
    double threshold;
    bool pin;
    double pinned_position;
};


class lattice{
    public:
    std::vector<Fiber> fiber;

    std::vector<std::mutex> mutexes;
    void add_fibers(double x,double y,double z,double pin);
    //int number_of_atoms(void){return static_cast<int>(atoms.size());}
};