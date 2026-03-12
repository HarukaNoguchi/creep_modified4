#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <random>
#include <omp.h> 
#include "fiber.hpp"
#include <string>
#include <sys/stat.h>
using namespace std;

void lattice::add_fibers(double r,double z,double f,double pin){
    Fiber a;
    a.z=z;
    a.f=f;
    a.threshold=r;
    a.pin=true;
    a.pinned_position=pin;
    fiber.push_back(a);//push_back(a)はvectorのメンバ関数
};
