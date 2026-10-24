// Strong finite-cavity baseline: harmonic correction / stopped-path coupling.
#include <cstdint>
#include <omp.h>
#include <memory>
#define WIH_QUERY_COUNTER 1
namespace {
struct alignas(64) QueryCount {std::uint64_t count[2]={0,0};};
QueryCount query_count[256];
inline void note_query(int k){++query_count[omp_get_thread_num()].count[k];}
void reset_queries(){for(auto& q:query_count)q=QueryCount{};}
std::uint64_t queries(int k){std::uint64_t n=0;for(auto q:query_count)n+=q.count[k];return n;}
struct Vec3;double custom_source(const Vec3&);
}
#define WIH_CUSTOM_SOURCE 1
#define main unused_compliance_entry
#include "../compiled_bunny/src/wih_twos.cpp"
#undef main
namespace {
Vec3 sensor_center;double sensor_radius;
struct Heat {Vec3 c;double r,a;};std::vector<Heat> heat;
double custom_source(const Vec3& x){double f=0;for(auto h:heat){double t=std::max(0.,1.-dot(x-h.c,x-h.c)/(h.r*h.r));f+=h.a*t*t;}return f;}
void load_config(const std::string& path){std::ifstream f(path);heat.clear();if(!(f>>sensor_center.x>>sensor_center.y>>sensor_center.z>>sensor_radius))throw std::runtime_error("bad config");Heat h;while(f>>h.c.x>>h.c.y>>h.c.z>>h.r>>h.a)heat.push_back(h);if(heat.empty())throw std::runtime_error("missing source");}
struct Value {double mean=0,se=0;std::uint64_t steps=0,caps=0,hits=0;};
}
int process(int argc,char** argv,std::unique_ptr<EmbreeMesh>& retained,std::string& path){
 try{
  std::vector<char*> args{argv[0]};std::string config;
  for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--config")config=argv[++i];else if(k=="--hot-only"||k=="--fd"||k=="--diff"){if(std::stoi(argv[++i])!=0)throw std::runtime_error("use zero legacy flags");}else args.push_back(argv[i]);}
  Options o=parse_options(int(args.size()),args.data());load_config(config);omp_set_num_threads(o.threads);reset_queries();auto start=std::chrono::steady_clock::now();
  if(!retained){retained=std::make_unique<EmbreeMesh>(o.mesh_path);path=o.mesh_path;}
  if(path!=o.mesh_path)throw std::runtime_error("geometry changed during session");
  auto sites=read_points(o.points_path),old=read_optional_points(o.cavities_path);std::vector<Value> values(sites.size());
  #pragma omp parallel for schedule(dynamic,1)
  for(std::int64_t i=0;i<std::int64_t(sites.size());++i){
   auto holes=old;holes.push_back(sites[i]);auto& v=values[i];double sum2=0;
   for(int n=0;n<o.walks;++n){
    auto key=splitmix64(o.seed)^splitmix64(i+734)^splitmix64(n+8131);std::mt19937_64 rng(splitmix64(key));std::uniform_real_distribution<double> U(0,1);
    double rad=std::cbrt(U(rng));Vec3 x=sensor_center+random_unit_vector(rng)*(sensor_radius*rad);double weight=std::pow(1-rad*rad,2)*35/8;
    auto outer=walk_to_boundary(*retained,holes,x,o,key+719);v.steps+=outer.steps;v.caps+=outer.hit_limit;double value=0;
    if(outer.hit.cavity_index==int(old.size())){
     ++v.hits;auto inner=walk(*retained,old,outer.hit.point,o,key+198377,false);v.steps+=inner.steps;v.caps+=inner.hit_limit;
     value=weight*inner.value/(4*3.141592653589793*o.cavity_radius);
    }
    v.mean+=value;sum2+=value*value;
   }
   v.mean/=o.walks;v.se=std::sqrt(std::max(0.,(sum2-o.walks*v.mean*v.mean)/(o.walks*(o.walks-1))));
  }
  double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::uint64_t steps=0,caps=0,hits=0;
  std::ofstream f(o.output_path);f<<"site,u,p,score,same_index,steps,caps,hits\n"<<std::setprecision(17);
  for(std::size_t i=0;i<values.size();++i){auto v=values[i];steps+=v.steps;caps+=v.caps;hits+=v.hits;f<<i<<",0,0,"<<v.mean<<','<<v.se<<','<<v.steps<<','<<v.caps<<','<<v.hits<<'\n';}
  if(caps)throw std::runtime_error("capped harmonic-correction walk");
  std::ofstream s(o.summary_path);s<<std::setprecision(17)<<"{\"total_seconds\":"<<sec<<",\"peak_rss_megabytes\":"<<peak_rss_megabytes()<<",\"distance_queries\":"<<queries(0)<<",\"parity_rays\":"<<queries(1)<<",\"steps\":"<<steps<<",\"caps\":"<<caps<<",\"hits\":"<<hits<<"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
int main(int argc,char** argv){
 std::unique_ptr<EmbreeMesh> mesh;std::string path;
 if(argc==2 && std::string(argv[1])=="--serve"){
  std::string line;while(std::getline(std::cin,line)){int n=std::stoi(line);if(n<=0||n>200)return 2;std::vector<std::string> v(n);for(auto& a:v)if(!std::getline(std::cin,a))return 2;std::vector<char*> args;for(auto& a:v)args.push_back(a.data());int code=process(n,args.data(),mesh,path);std::cout<<"DONE "<<code<<std::endl;if(code)return code;}return 0;
 }
 return process(argc,argv,mesh,path);
}
