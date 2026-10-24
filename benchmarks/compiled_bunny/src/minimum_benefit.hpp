// Included inside wih_twos.cpp's anonymous namespace.
// Conservative geometry/source arithmetic; confidence still assumes the ideal
// conditional random sampling laws stated in the manuscript.
#if defined(__FAST_MATH__)
#error Minimum-benefit validation must not use fast-math.
#endif

double down(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
double up(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
double positive_down(double x) { return std::max(0.0, down(x)); }
double coord(const Vec3 &v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }

struct CertBox { Vec3 lo, hi; };
CertBox merge_box(const CertBox &a, const CertBox &b) {
  return {{std::min(a.lo.x,b.lo.x),std::min(a.lo.y,b.lo.y),std::min(a.lo.z,b.lo.z)},
          {std::max(a.hi.x,b.hi.x),std::max(a.hi.y,b.hi.y),std::max(a.hi.z,b.hi.z)}};
}
double box_distance_lower(const CertBox &b, const Vec3 &x) {
  double sum = 0;
  for (int a=0;a<3;++a) {
    const double g=std::max({0.0,down(coord(b.lo,a)-coord(x,a)),down(coord(x,a)-coord(b.hi,a))});
    sum=positive_down(sum+positive_down(g*g));
  }
  return positive_down(std::sqrt(sum));
}
double point_distance_bound(const Vec3 &a,const Vec3 &b,bool upper) {
  double sum=0;
  for(int k=0;k<3;++k) {
    double d=std::abs(coord(a,k)-coord(b,k));
    d=upper?up(d):positive_down(d);
    const double q=upper?up(d*d):positive_down(d*d);
    sum=upper?up(sum+q):positive_down(sum+q);
  }
  return upper?up(std::sqrt(sum)):positive_down(std::sqrt(sum));
}

// A union of triangle bounding boxes contains the whole surface. Its distance
// is a lower bound on surface distance. No closest-triangle predicate is used.
class CertDistanceTree {
  struct Node { CertBox box; int left=-1,right=-1,begin=0,end=0; };
  std::vector<CertBox> boxes_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
  int build(int begin,int end) {
    const int index=static_cast<int>(nodes_.size());nodes_.emplace_back();
    CertBox box=boxes_[order_[begin]];
    for(int j=begin+1;j<end;++j)box=merge_box(box,boxes_[order_[j]]);
    nodes_[index].box=box;nodes_[index].begin=begin;nodes_[index].end=end;
    if(end-begin>8) {
      int axis=0;
      for(int a=1;a<3;++a)if(coord(box.hi,a)-coord(box.lo,a)>coord(box.hi,axis)-coord(box.lo,axis))axis=a;
      int mid=(begin+end)/2;
      std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,[&](int a,int b){
        return coord(boxes_[a].lo,axis)+coord(boxes_[a].hi,axis)<coord(boxes_[b].lo,axis)+coord(boxes_[b].hi,axis);
      });
      const int l=build(begin,mid),r=build(mid,end);nodes_[index].left=l;nodes_[index].right=r;
    }
    return index;
  }
  void visit(int id,const Vec3 &x,double &best) const {
    const Node &n=nodes_[id];
    if(box_distance_lower(n.box,x)>=best)return;
    if(n.left<0) {
      for(int j=n.begin;j<n.end;++j)best=std::min(best,box_distance_lower(boxes_[order_[j]],x));
    } else {
      const double dl=box_distance_lower(nodes_[n.left].box,x),dr=box_distance_lower(nodes_[n.right].box,x);
      if(dl<dr){if(dl<best)visit(n.left,x,best);if(dr<best)visit(n.right,x,best);}
      else{if(dr<best)visit(n.right,x,best);if(dl<best)visit(n.left,x,best);}
    }
  }
 public:
  explicit CertDistanceTree(const EmbreeMesh &mesh) {
    for(const Triangle &t:mesh.triangles()) {
      const Vec3 a=mesh.vertices()[t.a],b=mesh.vertices()[t.b],c=mesh.vertices()[t.c];
      boxes_.push_back(merge_box(merge_box({a,a},{b,b}),{c,c}));
    }
    order_.resize(boxes_.size());std::iota(order_.begin(),order_.end(),0);
    nodes_.reserve(2*boxes_.size());build(0,static_cast<int>(boxes_.size()));
  }
  double distance(const Vec3 &p) const {double best=std::numeric_limits<double>::infinity();visit(0,p,best);return best;}
};

double cert_domain_distance(const CertDistanceTree &tree,const Vec3 &p,
                           const std::vector<Vec3> &holes,double r) {
  double d=tree.distance(p);
  for(const Vec3 &c:holes)d=std::min(d,down(point_distance_bound(p,c,false)-r));
  return std::max(0.0,d);
}

// (1-t/n)^n <= exp(-t), n=1024; all positive operations rounded downward.
// Avoid relying on a correctly rounded transcendental exp implementation.
double exp_negative_lower(double t_upper) {
  double a=std::max(0.0,down(1.0-up(t_upper/1024.0)));
  for(int i=0;i<10;++i)a=positive_down(a*a);
  return a;
}
double cert_source_lower(const Vec3 &p,const Options &o) {
  if(o.source_mode=="affine") {
    if(o.source_y_slope<0)throw std::runtime_error("validator expects nonnegative affine slope");
    return std::max(0.0,down(1.0+down(o.source_y_slope*down(p.y+.5))));
  }
  const std::array<Vec3,3> centers{{{-.48,-.18,-.08},{.43,.19,-.07},{.02,-.31,-.02}}};
  const std::array<Vec3,3> widths{{{.24,.22,.16},{.21,.20,.15},{.30,.16,.19}}};
  const std::array<double,3> amplitude{{5.2,4.6,2.0}};
  double value=.08;
  for(int j=0;j<3;++j) {
    double exponent=0;
    for(int a=0;a<3;++a) {
      const double absdiff=up(std::abs(coord(p,a)-coord(centers[j],a)));
      const double q=up(absdiff/coord(widths[j],a));
      exponent=up(exponent+up(q*q));
    }
    value=down(value+positive_down(amplitude[j]*exp_negative_lower(exponent)));
  }
  return std::max(0.0,value);
}

struct CertSample {
  double value=0,terminal_distance=0;int steps=0;
  bool capped=false,clipped=false;std::uint64_t containment_failures=0;
};
CertSample cert_walk(const CertDistanceTree &tree,const std::vector<Vec3> &holes,
                    Vec3 point,const Options &o,std::uint64_t seed) {
  std::mt19937_64 rng(splitmix64(seed));CertSample out;
  for(int k=0;k<o.max_steps;++k) {
    const double d=cert_domain_distance(tree,point,holes,o.cavity_radius);
    out.terminal_distance=d;
    if(d<=o.epsilon)return out;
    const double s=std::min(o.validation_radius_cap,positive_down(.95*d));
    Vec3 y=point;
    if(o.source_mode=="housing")y=point+random_unit_vector(rng)*(s*beta_22(rng));
    const Vec3 next=point+random_unit_vector(rng)*s;
    if(!(point_distance_bound(point,next,true)<d) || !(point_distance_bound(point,y,true)<d)) {
      ++out.containment_failures;return out; // fail closed; never use this record
    }
    const double weight=positive_down(positive_down(s*s)/6.0);
    const double increment=positive_down(weight*cert_source_lower(y,o));
    out.value=positive_down(out.value+increment);++out.steps;
    if(out.value>=o.validation_value_cap){out.value=o.validation_value_cap;out.clipped=true;return out;}
    point=next;
  }
  out.capped=true;return out;
}

int run_minimum_benefit(const EmbreeMesh &mesh,const std::vector<Vec3> &holes,
                        const std::vector<Vec3> &points,const Options &o) {
  if(!std::numeric_limits<double>::is_iec559)throw std::runtime_error("IEEE binary64 required");
  const auto begin=std::chrono::steady_clock::now();CertDistanceTree tree(mesh);
  const double build_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  std::ofstream output(o.output_path),summary(o.summary_path);
  output<<std::setprecision(17)<<"site,sample,value,steps,capped,clipped\n";
  summary<<std::setprecision(17)<<"{\"scope\":\"conditional ideal-sampling certificate; conservative box geometry and source arithmetic\",\n"
    <<"\"distance_tree_build_seconds\":"<<build_seconds<<",\n\"distance_mode\":\"triangle_AABB_union_lower_bound\",\n"
    <<"\"sample_cap\":"<<o.validation_value_cap<<",\"radius_cap\":"<<o.validation_radius_cap
    <<",\"stages\":[\n";
  std::vector<int> stages;
  for(int n:std::vector<int>{1024,4096,16384,65536})if(n<o.walks)stages.push_back(n);
  stages.push_back(o.walks);bool first=true;std::uint64_t failures=0;
  for(std::size_t site=0;site<points.size();++site) {
    if(!mesh.inside(points[site]) || inside_cavity(points[site],holes,o.cavity_radius))
      throw std::runtime_error("candidate classified outside; exact driver predicate is also required");
    std::vector<CertSample> values(o.walks);int prev=0;double elapsed=0;
    for(int n:stages) {
      const auto start=std::chrono::steady_clock::now();
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for(int j=prev;j<n;++j) {
        const std::uint64_t key=splitmix64(o.seed)^splitmix64(site+0xa1224ULL)^splitmix64(j+0x6242cULL);
        values[j]=cert_walk(tree,holes,points[site],o,key);
      }
      elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      long double sum=0;std::uint64_t steps=0,clipped=0,capped=0;
      for(int j=0;j<n;++j){sum+=values[j].value;steps+=values[j].steps;clipped+=values[j].clipped;capped+=values[j].capped;}
      if(!first)summary<<",\n";first=false;
      summary<<"{\"site\":"<<site<<",\"n\":"<<n<<",\"mean\":"<<static_cast<double>(sum/n)
        <<",\"mean_steps\":"<<static_cast<double>(steps)/n<<",\"clipped\":"<<clipped<<",\"path_caps\":"<<capped
        <<",\"elapsed_seconds\":"<<elapsed<<",\"center_clearance_lower\":"<<cert_domain_distance(tree,points[site],holes,o.cavity_radius)
        <<",\"embree_center_distance\":"<<domain_distance(mesh,points[site],holes,o.cavity_radius)<<"}";
      for(int j=prev;j<n;++j){const auto &v=values[j];failures+=v.containment_failures;
        output<<site<<','<<j<<','<<v.value<<','<<v.steps<<','<<v.capped<<','<<v.clipped<<'\n';}
      prev=n;
    }
  }
  summary<<"\n],\"containment_failures\":"<<failures<<"}\n";
  if(failures)throw std::runtime_error("sample containment guard failed; validation record must be rejected");
  return 0;
}
