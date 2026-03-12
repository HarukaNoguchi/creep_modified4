#include "fiber.hpp"
#include <unordered_set>
#include <vector>
#include <omp.h> 

class MD {
private:
  lattice *lat;
  const int R=1;
  void makeini(int);
  void find_min(std::unordered_set<int>&,double&,double);
  void loading(double&,int&,std::ofstream&,int);
  void redistribute(std::unordered_set<int>&,std::unordered_set<int>&,int,double&);
  void fiber_breaking(std::unordered_set<int>&,std::unordered_set<int>&,double&,double);
  void loop(std::unordered_set<int>&,std::unordered_set<int>&,int,double&,double&);
  void convergence(void);
  void self_correlation(int,int,int,std::string);
  //void caliculate_fint(int;)
  void onestep(int,std::ofstream&);
  void twostep(int,std::ofstream&,std::ofstream&,int &);
  double Distance1(double,int,double&);
  double Distance2l(double,int,double&);
  double Distance2r(double,int,double&);

  double Distance3(double,int,double&);

  void h_renew(double&,double&);
  std::vector<double> d_seq_solver(const std::vector<int>& seg);

  void unstable_check(int);
  void unstable_check2(void);
  void Energy_gap_dist(std::ofstream&);


  //void unstable_check2(bool&,std::unordered_set<int>&);

  void deformation(int,std::ofstream&,int &);
  //void thermal_relaxation(std::string& str,int,std::ofstream&);
  void thermal_relaxation(int,std::ofstream&,int);
  void cluster_calc(std::vector<std::vector<double>>&,double &);

  public:
  MD(void);
  ~MD(void);
  void run(int);
  double C=1;
  bool next_check=true;
  double front_av;
  double front_av2;
  double f_ext;
  bool num_1;
  int N=256;
  double n_break;//壊れたボンドの数
  int seed;
  double gamma=1;
  double kappa=1;
  double G=10;
  double D=1;
  int n_redistributerd;
  double T=0.5;
  double dt=1e-1;
  double V=1e-6;
  double k=1;
  double omega=1;//pow(10,2);
  double n_act;
  int n_load;
  //double dx=0.1;
  double c=1;
  double C1=2*c;//25*c;
  double C2=c;//12.5*c;
  int break_check=0;


  //std::unordered_set<int> f_breaking;
  //std::unordered_set<int> f_ridistributed;
  std::vector<double> break_h;
  //std::vector<std::unordered_set<int>> f_breaking_private(omp_get_max_threads());
  //std::string filename2=filename1+"/pair"; 

  std::string map_filename="strain_map_";
  std::string s="stress_strain";
  std::string no_crack="crack_nonbreak";
  std::string final_crack="crack_final";
  std::string break_num="break_num";
  std::string stabile_num="stability";
};
