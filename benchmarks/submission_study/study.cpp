#ifndef WIH_POINT_CHUNK
#define WIH_POINT_CHUNK 4
#endif
// Reuse the existing geometry and multigrid operators without changing the
// retained compliance implementation. This executable has its own entry point.
#include <cstdint>
#include <omp.h>
#define WIH_QUERY_COUNTER 1
#define WIH_ENCLOSURE_FALLBACK 1
namespace {
struct alignas(64) QueryCount {std::uint64_t count[3]={0,0,0};};
QueryCount query_count[256];
inline void note_query(int k){++query_count[omp_get_thread_num()].count[k];}
void reset_queries(){for(auto& q:query_count)q=QueryCount{};}
std::uint64_t queries(int k){std::uint64_t n=0;for(auto q:query_count)n+=q.count[k];return n;}
}
#define WIH_CUSTOM_SOURCE 1
namespace { struct Vec3; double custom_source(const Vec3&); }
#define main wih_compliance_entry_point
#include "../compiled_bunny/src/wih_twos.cpp"
#undef main
#include <memory>

namespace {
Vec3 sensor_center{0.55, 0.18, -0.03};
double sensor_radius = 0.13;
struct Heat { Vec3 center; double radius, amplitude; };
std::vector<Heat> heat;
double bump(Vec3 x,Vec3 c,double r) {double t=std::max(0.,1.-dot(x-c,x-c)/(r*r));return t*t;}
double custom_source(const Vec3& x) {
#ifdef WIH_AFFINE_COMPLIANCE
 return 1.+.65*x.y;
#else
 double f=0;for(auto h:heat)f+=h.amplitude*bump(x,h.center,h.radius);return f;
#endif
}
void load_config(const std::string& path) {
 std::ifstream f(path); heat.clear();
 if(!(f>>sensor_center.x>>sensor_center.y>>sensor_center.z>>sensor_radius))throw std::runtime_error("bad sensor config");
 Heat h;while(f>>h.center.x>>h.center.y>>h.center.z>>h.radius>>h.amplitude)heat.push_back(h);
 if(heat.empty())throw std::runtime_error("missing compact heat sources");
}

double sensor_source(const Vec3& x) {
#ifdef WIH_AFFINE_COMPLIANCE
 return custom_source(x);
#else
  const double t = std::max(0.0, 1.0-dot(x-sensor_center,x-sensor_center)/
                                            (sensor_radius*sensor_radius));
  return t*t/(32*3.141592653589793*std::pow(sensor_radius,3)/105);
#endif
}
#ifdef WIH_ANALYTIC_SOURCE
// Newton potential of A*(1-|x-c|^2/R^2)^2_+ and exact centered-ball
// Green integral. Long double reduces cancellation near tangency.
long double bump_potential(long double d,long double R,long double A) {
  if(d>=R)return A*R*R*R*(8.L/105)/d;
  const auto t=d/R;return A*R*R*(1.L/6-t*t/6+t*t*t*t/10-t*t*t*t*t*t/42);
}
long double potential_primitive(long double d,long double R,long double A) {
  const auto t=std::min(d/R,1.L),t2=t*t,t4=t2*t2;
  const auto inner=A*R*R*R*R*(t2/12-t4/24+t4*t2/60-t4*t4/336);
  return inner+(d>R?A*R*R*R*(8.L/105)*(d-R):0);
}
double exact_bump_green(Vec3 x,double radius,Vec3 center,double support,double amplitude) {
  const long double d=norm(x-center),r=radius,R=support,A=amplitude;
  if(d>=r+R)return 0.;
  const auto at_center=bump_potential(d,R,A);
  long double mean;
  if(d+R<=r)mean=A*R*R*R*(8.L/105)/r;
  else if(d<1e-12L*R)mean=bump_potential(r,R,A);
  else mean=(potential_primitive(d+r,R,A)-potential_primitive(std::abs(d-r),R,A))/(2*d*r);
  const auto value=at_center-mean;
  if(value < -1e-11L*std::max(1.L,std::abs(at_center)))throw std::runtime_error("negative analytic Green integral");
  return double(std::max(0.L,value));
}
#endif
struct PairValue { double u=0, p=0, up=0, same=0; std::uint64_t steps=0, caps=0; };

PairValue paired_site(const EmbreeMesh& mesh, const std::vector<Vec3>& holes,
                     Vec3 x, std::size_t site, const Options& o, bool hot, int first_walk=0) {
  PairValue out;
  for (int n=first_walk; n<first_walk+o.walks; ++n) {
    int index=n;
#ifdef WIH_REPAIRED_BENCHMARK_STREAMS
    for(int seed:{1611,2711,3811,4911,6011})for(int offset:{0,8,32,128})
      if(n+571==seed+offset && n<4096)index=n+1000000;
#endif
    std::mt19937_64 rng(splitmix64(o.seed)^splitmix64(site+91347)^splitmix64(index+571));
    Vec3 y=x; double u=0,p=0;
    for (int k=0;k<o.max_steps;++k) {
      const double r=domain_distance(mesh,y,holes,o.cavity_radius);
      if (!(r>o.epsilon)) break;
      // Both source integrals share one Green-distributed point. Each entire
      // pair is independent under the usual ideal-PRNG sampling model.
#ifdef WIH_ANALYTIC_SOURCE
#ifdef WIH_AFFINE_COMPLIANCE
      const double contribution=r*r/6*custom_source(y);
      u+=contribution;if(!hot)p+=contribution;
#else
      for(auto h:heat)u+=exact_bump_green(y,r,h.center,h.radius,h.amplitude);
      if(!hot)p+=exact_bump_green(y,r,sensor_center,sensor_radius,
          105/(32*3.141592653589793*std::pow(sensor_radius,3)));
#endif
#else
      const Vec3 a=y+random_unit_vector(rng)*(r*beta_22(rng));
      u+=r*r/6*source(a,o);
      if (!hot) p+=r*r/6*sensor_source(a);
#endif
      y=y+random_unit_vector(rng)*r;
      ++out.steps;
      if(k+1==o.max_steps) ++out.caps;
    }
    out.u+=u;out.p+=p;out.same+=u*p;
  }
  const double n=o.walks;
  out.up=hot?out.u/n:(n>1?(out.u*out.p-out.same)/(n*(n-1)):0.);
  out.same/=n;out.u/=n;out.p/=n;
  return out;
}

#ifdef WIH_REPAIR_STREAMS
PairValue repair_site(const EmbreeMesh& mesh,const std::vector<Vec3>& holes,
                     Vec3 x,std::size_t site,const Options& o,bool hot) {
 PairValue out;Options one=o;one.walks=1;
 for(int seed:{1611,2711,3811,4911,6011})for(int offset:{0,8,32,128}) {
  const int n=seed+offset-571;if(n<0||n>=4096)continue;
  const auto old=paired_site(mesh,holes,x,site,one,hot,n);
  const auto replacement=paired_site(mesh,holes,x,site,one,hot,n+1000000);
  out.u+=replacement.u-old.u;out.p+=replacement.p-old.p;
  out.same+=replacement.same-old.same;
  out.steps+=old.steps+replacement.steps;out.caps+=old.caps+replacement.caps;
 }
 return out;
}
#endif

double interpolate(const MGLevel& a,Vec3 x) {
  const double c[3]={(x.x-a.low.x)/a.spacing[0],(x.y-a.low.y)/a.spacing[1],
                     (x.z-a.low.z)/a.spacing[2]};
  int b[3];double f[3];
  for(int d=0;d<3;++d) {const double v=std::clamp(c[d],0.,double(a.shape[d]-1));
    b[d]=std::min(int(std::floor(v)),a.shape[d]-2);f[d]=v-b[d];}
  double u=0;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int k=0;k<2;++k)
    u+=(i?f[0]:1-f[0])*(j?f[1]:1-f[1])*(k?f[2]:1-f[2])*
       a.u[grid_index(b[0]+i,b[1]+j,b[2]+k,a.shape)];
  return u;
}

std::vector<PairValue> mg_pair(const EmbreeMesh& mesh,const std::vector<Vec3>& holes,
    const std::vector<Vec3>& points,const Options& o,bool hot,double& residual) {
#ifdef WIH_REUSE_MG
  // Cache complete solved fields, not query values. Invalidate on every input
  // affecting the PDE, geometry, discretization, or algebraic stopping rule.
  static std::string cached_key;
  static MGLevel cached_u,cached_p;
  static double cached_residual=0;
  std::ostringstream key;key<<std::setprecision(17)<<o.mesh_path<<' '<<o.source_mode<<' '
    <<o.cavity_radius<<' '<<hot<<' '<<o.mg_cycles<<' '<<o.mg_tolerance
    <<' '<<o.mg_pre_smooth<<' '<<o.mg_post_smooth<<' '<<o.mg_coarse_smooth
    <<' '<<o.mg_omega<<' '<<o.source_y_slope;
  for(auto n:o.grid)key<<' '<<n;
  key<<' '<<sensor_center.x<<' '<<sensor_center.y<<' '<<sensor_center.z<<' '<<sensor_radius;
  for(auto h:heat)key<<' '<<h.center.x<<' '<<h.center.y<<' '<<h.center.z<<' '<<h.radius<<' '<<h.amplitude;
  for(auto h:holes)key<<' '<<h.x<<' '<<h.y<<' '<<h.z;
  if(key.str()==cached_key) {
    residual=cached_residual;
    std::vector<PairValue> out(points.size());
    #pragma omp parallel for
    for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
      if(inside_cavity(points[i],holes,o.cavity_radius))continue;
      out[i].u=interpolate(cached_u,points[i]);
#ifdef WIH_AFFINE_COMPLIANCE
      out[i].p=hot?0:out[i].u;
#else
      out[i].p=hot?0:interpolate(cached_p,points[i]);
#endif
      out[i].up=hot?out[i].u:out[i].u*out[i].p;
    }
    return out;
  }
  cached_key.clear();
#ifndef WIH_REUSE_OUTER
  cached_u=MGLevel{};cached_p=MGLevel{};
#endif
#endif
  std::vector<MGLevel> levels;auto shape=o.grid;
#ifdef WIH_REUSE_OUTER
  static std::array<int,3> outer_shape{0,0,0};
  static std::vector<MGLevel> outer;
  if(outer_shape!=o.grid){outer.clear();outer_shape=o.grid;}
#endif
  while(true) {
#ifdef WIH_REUSE_OUTER
    const auto l=levels.size();
    if(l==outer.size()) {
      auto geometry=make_mg_level(mesh,{},0.,shape);
      geometry.u.clear();geometry.rhs.clear();geometry.residual.clear();geometry.scratch.clear();
      geometry.u.shrink_to_fit();geometry.rhs.shrink_to_fit();geometry.residual.shrink_to_fit();geometry.scratch.shrink_to_fit();
      outer.push_back(std::move(geometry));
    }
    auto geometry=outer[l];const auto count=geometry.active.size();
    geometry.u.assign(count,0.);geometry.rhs.assign(count,0.);
    geometry.residual.assign(count,0.);geometry.scratch.assign(count,0.);
    if(!holes.empty()) {
      #pragma omp parallel for
      for(std::int64_t k=0;k<std::int64_t(count);++k) {
        if(!geometry.active[k])continue;
        const int iz=k%shape[2],iy=(k/shape[2])%shape[1],ix=k/(shape[2]*shape[1]);
        const Vec3 x{geometry.low.x+ix*geometry.spacing[0],geometry.low.y+iy*geometry.spacing[1],geometry.low.z+iz*geometry.spacing[2]};
        if(inside_cavity(x,holes,o.cavity_radius))geometry.active[k]=0;
      }
    }
    geometry.active_count=std::count(geometry.active.begin(),geometry.active.end(),static_cast<unsigned char>(1));
    levels.push_back(std::move(geometry));
#else
    levels.push_back(make_mg_level(mesh,holes,o.cavity_radius,shape));
#endif
    if(*std::min_element(shape.begin(),shape.end())<=9)break;
    shape={(shape[0]+1)/2,(shape[1]+1)/2,(shape[2]+1)/2};}
  std::vector<PairValue> out(points.size());
#ifdef WIH_AFFINE_COMPLIANCE
  const int fields=1;
#else
  const int fields=hot?1:2;
#endif
  for(int field=0;field<fields;++field) {
    for(auto& a:levels) {std::fill(a.u.begin(),a.u.end(),0.);
      std::fill(a.rhs.begin(),a.rhs.end(),0.);}
    auto& a=levels.front();double norm2=0;
#ifdef WIH_REUSE_OUTER
    const auto& previous=field?cached_p:cached_u;
    if(previous.shape==a.shape && previous.u.size()==a.u.size()) {
      #pragma omp parallel for
      for(std::int64_t k=0;k<std::int64_t(a.u.size());++k)
        a.u[k]=a.active[k]?previous.u[k]:0.;
    }
#endif
    #pragma omp parallel for collapse(3) reduction(+:norm2)
    for(int i=0;i<a.shape[0];++i)for(int j=0;j<a.shape[1];++j)for(int k=0;k<a.shape[2];++k) {
      const auto l=grid_index(i,j,k,a.shape);if(!a.active[l])continue;
      Vec3 x{a.low.x+i*a.spacing[0],a.low.y+j*a.spacing[1],a.low.z+k*a.spacing[2]};
      a.rhs[l]=field?sensor_source(x):source(x,o);norm2+=a.rhs[l]*a.rhs[l];
    }
    double rel=norm2>0?1:0;
    for(int c=0;c<o.mg_cycles && rel>o.mg_tolerance;++c) {
      mg_v_cycle(levels,0,o);rel=mg_residual_norm(a)/std::max(std::sqrt(norm2),1e-300);
    }
    residual=std::max(residual,rel);
    if(!std::isfinite(rel)||rel>o.mg_tolerance)throw std::runtime_error("MG did not reach requested residual");
#ifdef WIH_REUSE_MG
    auto& cache=field?cached_p:cached_u;
    cache.shape=a.shape;cache.low=a.low;cache.spacing=a.spacing;cache.u=a.u;
#endif
    #pragma omp parallel for
    for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
      if(inside_cavity(points[i],holes,o.cavity_radius))continue;
      if(field)out[i].p=interpolate(a,points[i]);else out[i].u=interpolate(a,points[i]);
    }
  }
#ifdef WIH_AFFINE_COMPLIANCE
  for(auto& v:out)v.p=hot?0:v.u;
#endif
#ifdef WIH_REUSE_MG
  cached_key=key.str();cached_residual=residual;
#endif
  for(auto& v:out)v.up=hot?v.u:v.u*v.p;
  return out;
}
}

int process_request(int argc,char** argv,std::unique_ptr<EmbreeMesh>& retained_mesh,
                    std::string& retained_path) {
  try {
    sensor_center={0.55,0.18,-0.03};
    bool hot=false,fd=false,diff=false;std::string config;std::vector<char*> args{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string key=argv[i];
      if(key=="--config") {config=argv[++i];}
      else if(key=="--fd") {fd=std::stoi(argv[++i])!=0;}
      else if(key=="--diff") {diff=std::stoi(argv[++i])!=0;}
      else if(key=="--sensor-side") {
        if(i+1==argc)throw std::runtime_error("missing sensor side");
        const int side=std::stoi(argv[++i]);
        if(side!=1&&side!=-1)throw std::runtime_error("sensor side must be -1 or 1");
        sensor_center={0.55*side,0.18*side,-0.03};
      } else if(key=="--hot-only") {
        if(i+1==argc)throw std::runtime_error("missing hot flag");
        hot=std::stoi(argv[++i])!=0;
      } else args.push_back(argv[i]);
    }
    const Options o=parse_options(int(args.size()),args.data());
    load_config(config);
    if(o.mode!="twos"&&o.mode!="mg")throw std::runtime_error("use twos or mg");
    if(o.threads>0)omp_set_num_threads(o.threads);
    reset_queries();
    const auto start=std::chrono::steady_clock::now();
    if(!retained_mesh) {retained_mesh=std::make_unique<EmbreeMesh>(o.mesh_path);retained_path=o.mesh_path;}
    if(retained_path!=o.mesh_path)throw std::runtime_error("a session must retain the same outer mesh");
    const EmbreeMesh& mesh=*retained_mesh;
    const auto points=read_points(o.points_path),holes=read_optional_points(o.cavities_path);
    std::vector<PairValue> values(points.size());double residual=0;
    if(diff) {
      std::vector<DifferentialSiteResult> ds(points.size());
      #pragma omp parallel for schedule(dynamic,4)
      for(std::int64_t i=0;i<std::int64_t(points.size());++i)
        ds[i]=evaluate_differential_site(mesh,holes,points[i],i,0,o);
      write_differential_field(o.output_path,points,ds,holes.size());
      std::uint64_t steps=0,unsafe=0,caps=0;
      for(auto& d:ds){steps+=d.outer_steps+d.nested_steps;unsafe+=d.unsafe_offset_terminations;caps+=d.limits;}
      if(unsafe||caps)throw std::runtime_error("unsafe differential path: unsafe="+std::to_string(unsafe)+" caps="+std::to_string(caps));
      double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::ofstream f(o.summary_path);f<<std::setprecision(17)<<"{\"total_seconds\":"<<sec<<",\"steps\":"<<steps<<",\"caps\":"<<caps<<",\"peak_rss_megabytes\":"<<peak_rss_megabytes()<<",\"distance_queries\":"<<queries(0)<<",\"parity_rays\":"<<queries(1)<<",\"enclosure_fallbacks\":"<<queries(2)<<"}\n";
      return 0;
    }
    if(fd) {
      #pragma omp parallel for schedule(dynamic,1)
      for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
        auto added=holes;added.push_back(points[i]);auto& v=values[i];double sum2=0;
        for(int n=0;n<o.walks;++n) {
          auto key=splitmix64(o.seed)^splitmix64(i+734)^splitmix64(n+8131);
          std::mt19937_64 rng(splitmix64(key));std::uniform_real_distribution<double> U(0,1);
          double rad=std::cbrt(U(rng));Vec3 x=sensor_center+random_unit_vector(rng)*(sensor_radius*rad);
          double weight=std::pow(1-rad*rad,2)*35/8;
          auto a=walk(mesh,holes,x,o,key+719,false),b=walk(mesh,added,x,o,key+719,false);
          double d=weight*(a.value-b.value)/(4*3.141592653589793*o.cavity_radius);
          v.up+=d;sum2+=d*d;v.steps+=a.steps+b.steps;v.caps+=a.hit_limit+b.hit_limit;
        }
        v.up/=o.walks;v.same=std::sqrt(std::max(0.,(sum2-o.walks*v.up*v.up)/(o.walks*(o.walks-1))));
      }
    }
    else if(o.mode=="mg")values=mg_pair(mesh,holes,points,o,hot,residual);
    else {
      #pragma omp parallel for schedule(dynamic,WIH_POINT_CHUNK)
      for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
        if(!inside_cavity(points[i],holes,o.cavity_radius))
#ifdef WIH_REPAIR_STREAMS
          values[i]=repair_site(mesh,holes,points[i],i,o,hot);
#else
          values[i]=paired_site(mesh,holes,points[i],i,o,hot);
#endif
      }
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::uint64_t steps=0,caps=0;
    std::ofstream f(o.output_path);f<<"site,u,p,score,same_index,steps,caps\n"<<std::setprecision(17);
    for(std::size_t i=0;i<values.size();++i) {const auto& v=values[i];steps+=v.steps;caps+=v.caps;
      f<<i<<','<<v.u<<','<<v.p<<','<<v.up<<','<<v.same<<','<<v.steps<<','<<v.caps<<'\n';}
    if(caps)throw std::runtime_error("walk step limit reached");
    std::ofstream s(o.summary_path);s<<std::setprecision(17)
      <<"{\"total_seconds\":"<<seconds<<",\"peak_rss_megabytes\":"<<peak_rss_megabytes()
      <<",\"distance_queries\":"<<queries(0)<<",\"parity_rays\":"<<queries(1)<<",\"enclosure_fallbacks\":"<<queries(2)<<",\"relative_residual\":"<<residual<<",\"steps\":"<<steps<<",\"caps\":"<<caps<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

#ifndef WIH_NO_STUDY_MAIN
int main(int argc,char** argv) {
  std::unique_ptr<EmbreeMesh> retained_mesh;std::string retained_path;
  if(argc==2 && std::string(argv[1])=="--serve") {
    // Length-prefixed argument lines preserve paths containing spaces. One
    // request is completed before the next arrives; no geometry decisions
    // are read ahead or omitted from the Python driver's wall clock.
    std::string line;
    while(std::getline(std::cin,line)) {
      int n=0;try{n=std::stoi(line);}catch(...){return 2;}
      if(n<=0||n>200)return 2;
      std::vector<std::string> values(n);
      for(auto& value:values)if(!std::getline(std::cin,value))return 2;
      std::vector<char*> args;for(auto& value:values)args.push_back(value.data());
      int code=process_request(n,args.data(),retained_mesh,retained_path);
      std::cout<<"DONE "<<code<<std::endl;
      if(code)return code;
    }
    return 0;
  }
  return process_request(argc,argv,retained_mesh,retained_path);
}

#endif
