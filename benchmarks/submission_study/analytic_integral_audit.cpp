#define WIH_POINT_CHUNK 1
#define WIH_ANALYTIC_SOURCE 1
#define WIH_NO_STUDY_MAIN 1
#include "study.cpp"
int main(){double d,r,R,A;std::cout<<std::setprecision(17);
 while(std::cin>>d>>r>>R>>A)std::cout<<exact_bump_green({d,0,0},r,{0,0,0},R,A)<<'\n';}
