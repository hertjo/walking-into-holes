// Reuse the existing geometry and multigrid operators without changing the
// retained compliance implementation. This executable has its own entry point.
#define main wih_compliance_entry_point
#include "../compiled_bunny/src/wih_twos.cpp"
#undef main
#include <memory>

namespace {
Vec3 sensor_center{0.55, 0.18, -0.03};
constexpr double sensor_radius = 0.13;
double sensor_source(const Vec3& x) {
  const double t = std::max(0.0, 1.0-dot(x-sensor_center,x-sensor_center)/
                                            (sensor_radius*sensor_radius));
  return t*t;
}
struct PairValue { double u=0, p=0, up=0, same=0; std::uint64_t steps=0, caps=0; };

PairValue paired_site(const EmbreeMesh& mesh, const std::vector<Vec3>& holes,
                     Vec3 x, std::size_t site, const Options& o, bool hot) {
  PairValue out;
  for (int n=0; n<o.walks; ++n) {
    std::mt19937_64 rng(splitmix64(o.seed)^splitmix64(site+91347)^splitmix64(n+571));
    Vec3 y=x; double u=0,p=0;
    for (int k=0;k<o.max_steps;++k) {
      const double r=domain_distance(mesh,y,holes,o.cavity_radius);
      if (!(r>o.epsilon)) break;
      // Both source integrals share one Green-distributed point. Each entire
      // pair is independent under the usual ideal-PRNG sampling model.
      const Vec3 a=y+random_unit_vector(rng)*(r*beta_22(rng));
      u+=r*r/6*source(a,o);
      if (!hot) p+=r*r/6*sensor_source(a);
      y=y+random_unit_vector(rng)*r;
      ++out.steps;
      if(k+1==o.max_steps) ++out.caps;
    }
    out.u+=u;out.p+=p;out.same+=u*p;
  }
  const double n=o.walks;
  out.up=hot?out.u/n:(out.u*out.p-out.same)/(n*(n-1));
  out.same/=n;out.u/=n;out.p/=n;
  return out;
}

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
  std::vector<MGLevel> levels;auto shape=o.grid;
  while(true) {levels.push_back(make_mg_level(mesh,holes,o.cavity_radius,shape));
    if(*std::min_element(shape.begin(),shape.end())<=9)break;
    shape={(shape[0]+1)/2,(shape[1]+1)/2,(shape[2]+1)/2};}
  std::vector<PairValue> out(points.size());
  for(int field=0;field<(hot?1:2);++field) {
    for(auto& a:levels) {std::fill(a.u.begin(),a.u.end(),0.);
      std::fill(a.rhs.begin(),a.rhs.end(),0.);}
    auto& a=levels.front();double norm2=0;
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
    if(rel>o.mg_tolerance)throw std::runtime_error("MG did not reach requested residual");
    #pragma omp parallel for
    for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
      if(inside_cavity(points[i],holes,o.cavity_radius))continue;
      if(field)out[i].p=interpolate(a,points[i]);else out[i].u=interpolate(a,points[i]);
    }
  }
  for(auto& v:out)v.up=hot?v.u:v.u*v.p;
  return out;
}
}

int process_request(int argc,char** argv,std::unique_ptr<EmbreeMesh>& retained_mesh,
                    std::string& retained_path) {
  try {
    sensor_center={0.55,0.18,-0.03};
    bool hot=false;std::vector<char*> args{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string key=argv[i];
      if(key=="--sensor-side") {
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
    if(o.mode!="twos"&&o.mode!="mg")throw std::runtime_error("use twos or mg");
    if(o.threads>0)omp_set_num_threads(o.threads);
    const auto start=std::chrono::steady_clock::now();
    if(!retained_mesh) {retained_mesh=std::make_unique<EmbreeMesh>(o.mesh_path);retained_path=o.mesh_path;}
    if(retained_path!=o.mesh_path)throw std::runtime_error("a session must retain the same outer mesh");
    const EmbreeMesh& mesh=*retained_mesh;
    const auto points=read_points(o.points_path),holes=read_optional_points(o.cavities_path);
    std::vector<PairValue> values(points.size());double residual=0;
    if(o.mode=="mg")values=mg_pair(mesh,holes,points,o,hot,residual);
    else {
      #pragma omp parallel for schedule(dynamic,4)
      for(std::int64_t i=0;i<std::int64_t(points.size());++i) {
        if(!inside_cavity(points[i],holes,o.cavity_radius))
          values[i]=paired_site(mesh,holes,points[i],i,o,hot);
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
      <<",\"relative_residual\":"<<residual<<",\"steps\":"<<steps<<",\"caps\":"<<caps<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

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
