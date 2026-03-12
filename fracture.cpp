#include <iostream>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sys/stat.h>
#include "MD.hpp"
#include "fiber.hpp"
#include <omp.h> 

using std::cout; using std::endl;
using std::system; using std::string;
int main(void){
    using namespace std;
    printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
        cout<<"creep time test"<<pow(10,8)<<endl;
        int t=0;
        #pragma omp parallel for 
        for (int seed=t;seed<t+44;seed++){
            MD *md = new MD();
            md->run(seed);
            delete md;
        }
        end = std::chrono::system_clock::now();  // 計測終了時間
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
        cout<<elapsed<<endl;

}