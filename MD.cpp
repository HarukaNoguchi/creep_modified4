#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <unordered_set>
#include <chrono>
#include <random>
#include <functional>
#include <omp.h>
#include "MD.hpp"
#include "fiber.hpp"
#include <filesystem>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <limits>
#include <set>

using namespace std;
double n_break;
std::vector<std::unordered_set<int>> f_breaking_private(omp_get_max_threads());    
std::vector<std::unordered_set<int>> f_ridistributed_private(omp_get_max_threads());
std::vector<std::vector<double>> h_old(omp_get_max_threads());
std::vector<std::vector<double>> g_old(omp_get_max_threads());

std::set<int> to_ordered_set(const std::unordered_set<int>& uset) {
    return std::set<int>(uset.begin(), uset.end());
}

#pragma omp threadprivate(n_break)

double monte_range(int seed,double min_val,double max_val)///熱活性化の判定に用いる
{
    // 乱数生成器
    thread_local static std::mt19937_64 mt64(seed);
    // [min_val, max_val] の一様分布整数 (real) の分布生成器
    std::uniform_real_distribution<> get_rand_uni_int(min_val, max_val);
    return get_rand_uni_int(mt64);
}


double get_rand_range(int seed,double min_val,double max_val)///破壊閾値を与える
{
    // 乱数生成器
    thread_local static std::mt19937_64 mt64(seed);
    // [min_val, max_val] の一様分布整数 (real) の分布生成器
    //std::uniform_real_distribution<> get_rand_uni_int(min_val, max_val);
    //return get_rand_uni_int(mt64);
    
    std::exponential_distribution<> dist(1/max_val);    
    return dist(mt64);
    
    /*
    std::normal_distribution<> dist(150, 20);
    for (int i=0;i<1000;i++){
        double a=dist(mt64);
        if (a>0){
            return (a);
            break;
        }
    }
    */

    //std::weibull_distribution<> dist2(0.5,5);
    //return dist2(mt64);
    //return 0.05;

}

double exponential_dist(int seed,double lambda){
    thread_local static std::mt19937_64 mt64(seed);
    std::exponential_distribution<> dist(lambda);    
    return dist(mt64);
}

double P_act(int seed,double lambda){
    static std::mt19937_64 mt64(seed);
    std::exponential_distribution<> dist(lambda);    
    return dist(mt64);
}

MD::MD(void)
{  lat = new lattice();
}
//------------------------------------------------------------------------
MD::~MD(void)
{delete lat;
}

void MD::makeini(int seed) {
    //std::ifstream ifs1("/home/noguchi/creep_modified/QEW_G_10_V_0.0000010000_N_1024_dt_0.1000000000_c_0.300000_G_10.000000_D_1.000000/T_3.000000/front_seed1.dat");
    std::ostringstream oss;
    int  k=seed;

    // 初期状態を平坦な状態から始める場合
    
    for (int i=0;i<N;++i){
        f_ext=0;
        double r;
        r=get_rand_range(seed,0,G);
        (lat->add_fibers)(r,0,0,exponential_dist(seed,1));
    }
}

void MD::unstable_check(int t){
    for (int i=0;i<N;i++){
        if (((lat->fiber)[i].threshold)-(f_ext- ((lat->fiber)[i].z))-((lat->fiber)[i].f)<0){
            f_breaking_private[omp_get_thread_num()].insert(i);
        }
        else if (((lat->fiber)[i].pin)==false){
            f_breaking_private[omp_get_thread_num()].insert(i);
        }
    }
}

void MD::unstable_check2(){
    for (int i=0;i<N;i++){
        if (((lat->fiber)[i].threshold)-(f_ext- ((lat->fiber)[i].z))-((lat->fiber)[i].f)<=0){
            //n_act+=1;
            f_breaking_private[omp_get_thread_num()].insert(i);
            //num_1=true;
        }
    }
}


///両端は動かず、単一セグメントのみ活性化
double MD::Distance1(double h,int i,double& d){
    double K=pow((1+pow((((lat->fiber)[i].z)-(lat->fiber)[((i+1)%N)].z),2)),1.5);
    double A=(3*((((lat->fiber)[i].z)-(lat->fiber)[((i+1)%N)].z)))/(1+pow((((lat->fiber)[i].z)-(lat->fiber)[((i+1)%N)].z),2));

    double h_stop=((f_ext-((lat->fiber)[i].z))+(c/K)*(lat->fiber)[((i+1)%N)].z+ (lat->fiber)[((i-1)%N+N)%N].z)/(1+2*K-A*K*(((lat->fiber)[((i+1)%N)].z)+((lat->fiber)[((i-1)%N)].z)-2*((lat->fiber)[(i)].z)));
    if (h_stop>=h){
        return abs(h);
        d+=abs(h);
    }
    else{
        h_stop=std::max(h_stop,1e-5);
        return h_stop;//abs(h_stop);
    }
}

///連続した二つのセグメントが活性化
double MD::Distance2l(double h,int i,double& d){
    double K=pow((1+pow((((lat->fiber)[i].z)-(lat->fiber)[((i+1)%N)].z),2)),1.5);
    double c1=c/K;
    double term2=c1*((lat->fiber)[((i+1)%N)].z+(lat->fiber)[((i-1)%N+N)%N].z-2*((lat->fiber)[i].z));
    double term3=c1*(f_ext-(lat->fiber)[((i+1)%N)].z)/(1+2*c);
    double term4=c1*c1*((lat->fiber)[((i+2)%N)].z+(lat->fiber)[i].z-2*((lat->fiber)[((i+1)%N)].z))/(1+2*c1);
    double h_stop=((f_ext-((lat->fiber)[i].z))+term2+term3-term4)/(1+2*c1+c1*c1/(1+2*c1));
    
    if (h_stop>=h){
        return abs(h);
        d+=abs(h);
    }
    else{
        h_stop=std::max(h_stop,1e-2);
        return h;//abs(h_stop);
    }

}

double MD::Distance2r(double h,int i,double& d){
    double K=pow((1+pow((((lat->fiber)[i].z)-(lat->fiber)[((i+1)%N)].z),2)),1.5);
    double c1=c/K;

    double term2=c1*((lat->fiber)[((i+2)%N)].z+(lat->fiber)[((i)%N+N)%N].z-2*((lat->fiber)[i+1].z));
    double term3=c1*(f_ext-(lat->fiber)[((i)%N)].z)/(1+2*c1);
    double term4=c1*c1*((lat->fiber)[((i+1)%N)].z+(lat->fiber)[i-1].z-2*((lat->fiber)[((i)%N)].z))/(1+2*c1);
    double h_stop=((f_ext-((lat->fiber)[i].z))+term2+term3-term4)/(1+2*c1+c1*c1/(1+2*c1));
    
    if (h_stop>=h){
        return abs(h);
        d+=abs(h);
    }
    else{
        h_stop=std::max(h_stop,1e-2);
        return abs(h_stop);
    }

}


///連続した3つ以上のセグメントが活性化したさい、その内部
double MD::Distance3(double h,int i,double& d){
    double h_stop=((f_ext+((lat->fiber)[i].z))-(lat->fiber)[((i+1)%N)].z- (lat->fiber)[((i-1)%N+N)%N].z)/(1+2*c);
    if (h_stop>=h){
        return h;
        d+=h;
    }
    else{
        return h_stop;
    }
}

int max_consecutive_segment_len_periodic(const std::unordered_set<int>& S, int N) {
    if (S.empty()) return 0;
    if (N <= 0) return 0;

    // セグメントの「始点」: x ∈ S かつ prev=(x-1)∉S
    int maxlen = 1;
    int n_starts = 0;

    for (int x : S) {
        int prev = (x - 1 + N) % N;
        if (!S.count(prev)) {
            n_starts++;

            // 始点から forward に伸ばして長さを数える
            int len = 1;
            int cur = x;
            while (true) {
                int next = (cur + 1) % N;
                if (!S.count(next)) break;
                cur = next;
                len++;

                // 念のため：無限ループ防止（理論上は不要だが安全）
                if (len > (int)S.size()) break;
            }
            maxlen = std::max(maxlen, len);
        }
    }

    // 始点が 0 個なら、S は「輪」状の1セグメント（例: {N-2,N-1,0,1} など）
    if (n_starts == 0) {
        return (int)S.size();
    }

    return maxlen;
}


#include <unordered_set>
#include <vector>
#include <algorithm>

std::vector<std::vector<int>>
split_into_segments_periodic(const std::unordered_set<int>& S, int N)
{
    std::vector<std::vector<int>> segments;
    if (S.empty()) return segments;
    if (N <= 0) return segments;

    // 1) unordered_set -> vector にしてソート
    std::vector<int> v;
    v.reserve(S.size());
    for (int x : S) {
        int y = ((x % N) + N) % N;  // 念のため [0,N-1] に正規化
        v.push_back(y);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    // 2) 非周期として一旦セグメント分割
    std::vector<std::vector<int>> temp;
    std::vector<int> cur;
    cur.push_back(v[0]);

    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] == v[i - 1] + 1) {
            cur.push_back(v[i]);
        } else {
            temp.push_back(std::move(cur));
            cur.clear();
            cur.push_back(v[i]);
        }
    }
    temp.push_back(std::move(cur));

    // 3) 周期境界チェック
    //    先頭が 0 を含み、末尾が N-1 を含むなら結合
    if (temp.size() >= 2 &&
        !temp.front().empty() &&
        !temp.back().empty() &&
        temp.front().front() == 0 &&
        temp.back().back() == N - 1)
    {
        std::vector<int> merged = temp.back();
        merged.insert(merged.end(), temp.front().begin(), temp.front().end());

        segments.push_back(std::move(merged));
        for (size_t i = 1; i + 1 < temp.size(); ++i) {
            segments.push_back(std::move(temp[i]));
        }
    } else {
        segments = std::move(temp);
    }

    return segments;
}

std::vector<double> thomas_solve(
    const std::vector<double>& a,  // sub-diagonal, size n-1
    const std::vector<double>& b,  // diagonal, size n
    const std::vector<double>& c,  // super-diagonal, size n-1
    const std::vector<double>& d   // RHS, size n
) {
    int n = (int)b.size();
    if (n == 0) return {};
    if ((int)a.size() != n - 1 || (int)c.size() != n - 1 || (int)d.size() != n) {
        throw std::runtime_error("thomas_solve: size mismatch");
    }

    if (n == 1) {
        if (std::abs(b[0]) < 1e-14) {
            throw std::runtime_error("thomas_solve: zero pivot");
        }
        return { d[0] / b[0] };
    }

    std::vector<double> cp(n - 1);
    std::vector<double> dp(n);
    std::vector<double> x(n);

    double denom = b[0];
    if (std::abs(denom) < 1e-14) {
        throw std::runtime_error("thomas_solve: zero pivot at 0");
    }

    cp[0] = c[0] / denom;
    dp[0] = d[0] / denom;

    for (int i = 1; i < n; ++i) {
        denom = b[i] - a[i - 1] * cp[i - 1];
        if (std::abs(denom) < 1e-14) {
            throw std::runtime_error("thomas_solve: zero pivot");
        }

        if (i < n - 1) {
            cp[i] = c[i] / denom;
        }
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }

    return x;
}


std::vector<double> MD::d_seq_solver(const std::vector<int>& seg)
{
    //cout<<seg.size()<<"hello"<<endl;
    const int L = (int)seg.size();
    if (L == 0) return {};

    // 1点だけなら Distance1 に任せてもよいが、
    // ここでは連立の一般形でそのまま処理可能
    const double alpha = c;          // 隣接結合
    const double beta  = 1.0 + 2.0*c; // 対角成分

    std::vector<double> a(std::max(0, L - 1), -alpha); // sub
    std::vector<double> b(L, beta);                    // diag
    std::vector<double> cc(std::max(0, L - 1), -alpha); // super
    std::vector<double> rhs(L, 0.0);

    for (int j = 0; j < L; ++j) {
        int idx = seg[j];

        // セグメント順に左・右を見る
        int idx_left;
        int idx_right;

        if (j == 0) {
            // セグメント外の左隣
            idx_left = (idx - 1 + N) % N;
        } else {
            idx_left = seg[j - 1];
        }

        if (j == L - 1) {
            // セグメント外の右隣
            idx_right = (idx + 1) % N;
        } else {
            idx_right = seg[j + 1];
        }

        double h  = (lat->fiber)[idx].z;
        double hl = (lat->fiber)[idx_left].z;
        double hr = (lat->fiber)[idx_right].z;

        // 右辺:
        // (1+2c)d_j - c d_{j-1} - c d_{j+1}
        //   = f_ext - h_j + c(h_{j-1}+h_{j+1}-2h_j)
        rhs[j] = f_ext - h + c * (hl + hr - 2.0 * h);
    }

    std::vector<double> d_seq = thomas_solve(a, b, cc, rhs);
    for (int i=0;i<d_seq.size();i++){
        //cout<<"answer"<<d_seq[i]<<endl;
    }
    // 非負制約を入れたいならここでクリップ
    for (double& x : d_seq) {
        if (x <= 0.0) x = 0.01;
    }

    return d_seq;
}



void MD::deformation(int seed,std::ofstream& ofs, int& check){
    auto& S = f_breaking_private[omp_get_thread_num()];
    auto segs = split_into_segments_periodic(S, N);  
    double step=0;
    ///一体のみ　単純なもの
    for (auto &seg : segs){
        int i=seg[0];
        double h=((lat->fiber)[i].pinned_position - (lat->fiber)[i].z) ;
        double step=0;
        double d = Distance1(h, i,step);     
        
        (lat->fiber)[i].z+=d;
        (lat->fiber)[i].f-=C1*d;
        (lat->fiber)[((i+1)%N)].f+=C2*d;        
        (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;

        //////Judgeing for pin->unpin or unpin->pin or unpin, and renewing fracture energy 
       ////ピン止めされていたもの
        if ((lat->fiber)[i].pinned_position-(lat->fiber)[i].z<1e-6 || (lat->fiber)[i].z>(lat->fiber)[i].pinned_position)  //進展した結果、ピン止めされている
        {
            (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
            (lat->fiber)[i].pin=true;
            (lat->fiber)[i].pinned_position=std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+exponential_dist(seed, 1/D);
        }
        else{
            (lat->fiber)[i].threshold=0;
            (lat->fiber)[i].pin=false;
        }
        n_break += step;
    }
    ///
            for (auto &seg : segs){
            if(seg.size()==1){
                int i=seg[0];
                double h=abs((lat->fiber)[i].pinned_position - (lat->fiber)[i].z) ;
                double d = 0.0;
                double step=0;
                d = Distance1(h, i,step);     
                
                (lat->fiber)[i].z+=d;
                (lat->fiber)[i].f-=C1*d;
                (lat->fiber)[((i+1)%N)].f+=C2*d;        
                (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;

                //////Judgeing for pin->unpin or unpin->pin or unpin, and renewing fracture energy 
                if((lat->fiber)[i].pin==true){
                    if ((lat->fiber)[i].pinned_position==(lat->fiber)[i].z){ //進展した結果、ピン止めされている
                        (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                        (lat->fiber)[i].pin=true;
                        (lat->fiber)[i].pinned_position=std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+exponential_dist(seed, 1/D);
                    }
                    else{
                        (lat->fiber)[i].threshold=0;
                        (lat->fiber)[i].pin=false;
                    }
                }
                else{
                    if ((lat->fiber)[i].pinned_position-(lat->fiber)[i].z<1e-3){//ピン止め一の周辺にいる
                        (lat->fiber)[i].pin=true;
                        (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                        (lat->fiber)[i].pinned_position=std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+exponential_dist(seed,1/D);
                    }
                    else{
                        (lat->fiber)[i].threshold=0;
                        (lat->fiber)[i].pin=false;
                    }
                }
                n_break += step;
            }
            else if(seg.size()==2){
                double h1=abs((lat->fiber)[seg[0]].pinned_position - (lat->fiber)[seg[0]].z) ;
                double h2=abs((lat->fiber)[seg[1]].pinned_position - (lat->fiber)[seg[1]].z) ;
                double d1 = Distance2l(h1,seg[0],step);     
                double d2 = Distance2r(h2,seg[1],step);
                
                if (h1==d1 && h2==d2){
                    // 何もしない
                }
                else if (h1!=d1 && h2!= d2){
                    // 何もしない
                }
                else if (h1!=d1){
                    d1=Distance1(h1,seg[0],step);
                }
                else if (h2!=d2){
                    d1=Distance1(h2,seg[1],step);
                }
                //// 決定した進展距離に基づく処理
                for (auto &i : seg){
                    double d;
                    if (i==0){
                        d=d1;
                    }
                    else{
                        d=d2;
                    }
                    (lat->fiber)[i].z+=d;
                    (lat->fiber)[i].f-=C1*d;
                    (lat->fiber)[((i+1)%N)].f+=C2*d;        
                    (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;

                    if((lat->fiber)[i].pin==true){
                        if ((lat->fiber)[i].pinned_position-(lat->fiber)[i].z<1e-4){ //進展した結果、ピン止めされている
                            (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                            (lat->fiber)[i].pin=true;
                            (lat->fiber)[i].pinned_position=std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+exponential_dist(seed, 1/D);
                        }
                        else{
                            (lat->fiber)[i].threshold=0;
                            (lat->fiber)[i].pin=false;
                        }
                    }
                    else{
                        if ((lat->fiber)[i].pinned_position-(lat->fiber)[i].z<1e-3 || (lat->fiber)[i].pinned_position<(lat->fiber)[i].z){//ピン止め一の周辺にいる
                            (lat->fiber)[i].pin=true;
                            (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                            (lat->fiber)[i].pinned_position=std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+exponential_dist(seed,1/D);
                        }
                        else{
                            (lat->fiber)[i].threshold=0;
                            (lat->fiber)[i].pin=false;
                        }
                    }
                    n_break += step;
                }
            }
            else {
                int L = seg.size();
                std::vector<double> d_seq = d_seq_solver(seg);
                std::unordered_set<int> S_split; 
                for (int j = 0; j < L; ++j) {
                    int i = seg[j];
                    double h_pin = ((lat->fiber)[i].pinned_position - (lat->fiber)[i].z);
                    if (h_pin<0){
                        cout<<"error"<<endl;
                    }
                    double d = d_seq[j];
                    
                    // まずは単純に min(平衡進展距離, pin距離) を採用
                    if (d > h_pin){
                        d = h_pin;
                        (lat->fiber)[i].z += d;
                        (lat->fiber)[i].f -= C1 * d;
                        (lat->fiber)[((i + 1) % N)].f += C2 * d;
                        (lat->fiber)[((i - 1 + N) % N)].f += C2 * d;
                        n_break += d;
                    }
                    else{
                        S_split.insert(i);
                    }
                }
                if (S_split.size()!=0){
                    auto re_segs = split_into_segments_periodic(S_split, N);
                    for (auto &re_seg : re_segs){
                        if(re_seg.size()==1){
                            int i=re_seg[0];
                            double h=abs((lat->fiber)[i].pinned_position - (lat->fiber)[i].z) ;
                            double d = 0.0;
                            double step=0;
                            d = Distance1(h, i,step);     
                            (lat->fiber)[i].z+=d;
                            (lat->fiber)[i].f-=C1*d;
                            (lat->fiber)[((i+1)%N)].f+=C2*d;        
                            (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;
                        }    
                        else if(re_seg.size()==2){
                            double h1=abs((lat->fiber)[re_seg[0]].pinned_position - (lat->fiber)[re_seg[0]].z) ;
                            double h2=abs((lat->fiber)[re_seg[1]].pinned_position - (lat->fiber)[re_seg[1]].z) ;
                            double d1 = Distance2l(h1,re_seg[0],step);     
                            double d2 = Distance2r(h1,re_seg[1],step);
                            //// 決定した進展距離に基づく処理
                            for (int j=0;j<2;j++){
                                int i=re_seg[j];
                                double d=(j==0 ?d1:d2 );
                                (lat->fiber)[i].z+=d;
                                (lat->fiber)[i].f-=C1*d;
                                (lat->fiber)[((i+1)%N)].f+=C2*d;        
                                (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;
                            }
                        }
                        else{
                            int LL = re_seg.size();
                            std::vector<double> d_seq2 = d_seq_solver(re_seg);
                            std::unordered_set<int> S_split2;

                            for (int j = 0; j < LL; ++j) {
                                int i = re_seg[j];
                                double h_pin = (lat->fiber)[i].pinned_position - (lat->fiber)[i].z;
                                double d = d_seq2[j];

                                // pin位置の方が先なら、pinで止める
                                
                                (lat->fiber)[i].z += d;
                                (lat->fiber)[i].f -= C1 * d;
                                (lat->fiber)[((i + 1) % N)].f += C2 * d;
                                (lat->fiber)[((i - 1 + N) % N)].f += C2 * d;
                                n_break += d;
                            }
                        }
                    }
                }
                for (int j = 0; j < L; ++j){ /////pin状態の更新
                    int i = seg[j];
                    if ((lat->fiber)[i].pin == true) {
                        if (std::abs((lat->fiber)[i].pinned_position - (lat->fiber)[i].z)<1e-3) {
                            (lat->fiber)[i].threshold = get_rand_range(seed, 0, G);
                            (lat->fiber)[i].pin = true;
                            (lat->fiber)[i].pinned_position =std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+ exponential_dist(seed, 1);
                        }
                        else {
                            (lat->fiber)[i].threshold = 0;
                            (lat->fiber)[i].pin = false;
                        }
                    }
                    else {
                        if (((lat->fiber)[i].pinned_position - (lat->fiber)[i].z < 1e-3) || ((lat->fiber)[i].pinned_position < (lat->fiber)[i].z) ) {
                            (lat->fiber)[i].pin = true;
                            (lat->fiber)[i].threshold = get_rand_range(seed, 0, G);
                            (lat->fiber)[i].pinned_position =std::max((lat->fiber)[i].z,(lat->fiber)[i].pinned_position)+ exponential_dist(seed, 1);
                        }
                        else {
                            (lat->fiber)[i].threshold = 0;
                            (lat->fiber)[i].pin = false;
                        }
                    }
                }
            }
    
        }
    }        


/*
    void MD::deformation(int seed,std::ofstream& ofs, int& check){
    auto& S = f_breaking_private[omp_get_thread_num()]; // std::set<int>
    int max_len = max_consecutive_segment_len_periodic(S, N);    
    if (max_len==1){
        for (int i : S) {
            double h=abs((lat->fiber)[i].pinned_position - (lat->fiber)[i].z) ;
            double d = 0.0;
            double step=0;
            d = Distance1(h, i,step);     
            
            (lat->fiber)[i].z+=d;
            (lat->fiber)[i].f-=C1*d;
            (lat->fiber)[((i+1)%N)].f+=C2*d;        
            (lat->fiber)[((i-1)%N+N)%N].f+=C2*d;

            //////Judgeing for pin->unpin or unpin->pin or unpin, and renewing fracture energy 
            if((lat->fiber)[i].pin==true){
                if ((lat->fiber)[i].pinned_position==(lat->fiber)[i].z){ //進展した結果、ピン止めされている
                    (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                    (lat->fiber)[i].pin=true;
                    (lat->fiber)[i].pinned_position+=exponential_dist(seed, 1/D);
                }
                else{
                    (lat->fiber)[i].threshold=0;
                    (lat->fiber)[i].pin=false;
                }
            }
            else{
                if ((lat->fiber)[i].pinned_position-(lat->fiber)[i].z<1e-6){//ピン止め一の周辺にいる
                    (lat->fiber)[i].pin=true;
                    (lat->fiber)[i].threshold=get_rand_range(seed,0,G);
                    //(lat->fiber)[i].pinned_position=(lat->fiber)[i].z+exponential_dist(seed,1/D);
                    (lat->fiber)[i].pinned_position+=exponential_dist(seed,1/D);
                }
                else{
                    (lat->fiber)[i].threshold=0;
                    (lat->fiber)[i].pin=false;
                }
            }
            n_break += step;
        }
    }        
}
*/

void MD::thermal_relaxation(int seed,std::ofstream& ofs,int w){
    for (int i=0;i<N;i++){
        if ((lat->fiber)[i].threshold!=0 && (((lat->fiber)[i].pin))==true){
            double p_i=dt*exp( -(((lat->fiber)[i].threshold) - (f_ext-((lat->fiber)[i].z))-((lat->fiber)[i].f))/T);
            double x=(((lat->fiber)[i].threshold) - (f_ext-((lat->fiber)[i].z))-((lat->fiber)[i].f));
            double r=monte_range(seed,0.0,1.0);
            if(r<p_i){
                n_act+=1;
                f_breaking_private[omp_get_thread_num()].insert(i);
            }
        }
    }
}

void MD::onestep(int seed,std::ofstream& f){///for mechanical loading
    for(int t=0;t < 100000;++t){
        unstable_check(num_1);
        if (f_breaking_private[omp_get_thread_num()].size()>0){
            deformation(seed,f,n_redistributerd);
            num_1=false;
        }
        else{
            break;
        }
        f_breaking_private[omp_get_thread_num()].clear();
        if (t==pow(10,5)-1){
            cout<<"out"<<" "<<n_redistributerd<<endl;
        }
    }
}

void MD::twostep(int seed,std::ofstream& f,std::ofstream& f1, int& n_redistributerd){///for thermal fluctuation
    for(int t=0;t<pow(10,8);++t){
        unstable_check2();
        if (f_breaking_private[omp_get_thread_num()].size()>0){
            //std::set<int> oset = to_ordered_set(f_breaking_private[omp_get_thread_num()]);
            if (t==0){
                deformation(seed,f1,n_redistributerd);
            }
            else{
                deformation(seed,f,n_redistributerd);
            }
            num_1=false;
        }
        else{
            break;
        }

        f_breaking_private[omp_get_thread_num()].clear();
        
        if (t==pow(10,8)-1){
            cout<<"out"<<" "<<n_redistributerd<<endl;
        }
    }
}

std::string to_string_with_precision(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void MD::Energy_gap_dist(std::ofstream& ofs){
    for(int i=0;i<N;i++){
        ofs<< (((lat->fiber)[i].threshold) - (f_ext-((lat->fiber)[i].z))-((lat->fiber)[i].f))<<endl;
    }
}

//vec_in,vec_outをつくる
void MD::run(int seed)
{   
    ///make file///
    std::string filename1="./TEST2_acc_G_20_V_"+to_string_with_precision(V, 10) +"_N_"+std::to_string(N)+"_dt_"+to_string_with_precision(dt, 10)+"_c_"+std::to_string(c)+"_G_"+std::to_string(G)+"_D_"+std::to_string(D);
    cout<<filename1<<endl;
    
    mkdir((filename1).c_str(),0777);
    
    std::string filename2=filename1+"/T_"+std::to_string(T);
    mkdir((filename2).c_str(),0777);

    string SD_all2 = "./" + filename2 + "/SD_all2_"+std::to_string(seed)+".dat";
    //ofstream ofs_SD_all2_damy(SD_all2);

    string SD_all = "./" + filename2 + "/SD_all_"+std::to_string(seed)+".dat";
    //ofstream ofs_SD_all_damy(SD_all);

    string SD_ini = "./" + filename2 + "/SD_ini_"+std::to_string(seed)+".dat";
    //ofstream ofs_SD_ini_damy(SD_ini);   

    string energy_gap2 = "./" + filename2 + "/tortal_energy_barrier"+std::to_string(seed)+".dat";
    ofstream ofs_energy_gap2(energy_gap2);

    string velocity = "./" + filename2 + "/_velocity_"+std::to_string(seed)+".dat";
    ofstream ofs_velocity(velocity);


    makeini(seed);
    int t=0;    
    double min=0;
    int Break_times=0;
    double v=V;//V+seed*1e-3/(36*3);
    n_redistributerd=0;
    ofstream ofs_SD_all2(SD_all2);
    ofstream ofs_SD_all(SD_all);
    ofstream ofs_SD_ini(SD_ini);   
    for (int w=0;w<1e7;w++){
        f_ext+=v*dt;
        onestep(seed,ofs_SD_all);
        thermal_relaxation(seed,ofs_energy_gap2,w);
        twostep(seed,ofs_SD_all,ofs_SD_ini,n_redistributerd);
    }

    bool on=false;
    int count=0;
    bool pin_state=(lat->fiber)[0].pin;
    for (int w=0;w<1e9;++w){
        //v=V/(pow(abs(1-(double(w)/1.2e8)),(2)));
        if (v>1e-1){
            cout<<v<<" "<<"break"<<endl;
            break;
        }

        n_break=0;
        n_act=0;
        n_load=0;
        f_ext+=v*dt;
        onestep(seed,ofs_SD_all);
        thermal_relaxation(seed,ofs_energy_gap2,w);
        twostep(seed,ofs_SD_all,ofs_SD_ini,n_redistributerd);
        /*
        if (pin_state!=(lat->fiber)[0].pin && seed==0){
            cout<<w<<" "<<(lat->fiber)[0].pin<<endl;
            pin_state=(lat->fiber)[0].pin;
        }
        */

        if (n_act>0){
            on=true;
            double h_av=0;
            double min=1000;
            int num=0;
            for (int i=0;i<N;i++){
                h_av+=(lat->fiber)[i].z;
            }
            ofs_velocity<<w<<" "<<v<<" "<<n_break<<" "<<f_ext<<" "<<n_load<<" "<<n_act<<" "<<n_redistributerd<<endl;
            Break_times+=1;
            count+=1;
        }
        h_old[omp_get_thread_num()].clear();
        g_old[omp_get_thread_num()].clear();
    }

    string front = "./" + filename2 + "/front_seed"+std::to_string(seed)+".dat";

    ofstream ofs_front(front);
    ofs_front<<"index"<<" "<<"z"<<" "<<"pin_position"<<" "<<"th"<<" "<<"n_redistributerd"<<" "<<"distance_for_pin_position"<<" "<<"pinned_or_not"<<endl;
    for(int i=0;i<N;i++){
        ofs_front<<std::setprecision(20)<<i<<" "<<(lat->fiber)[i].z<<" "<<(lat->fiber)[i].pinned_position<<" "<<(lat->fiber)[i].threshold<<" "<<n_redistributerd<<" "<<(lat->fiber)[i].pinned_position-(lat->fiber)[i].z<<" "<<(lat->fiber)[i].pin<<endl;
    }

};