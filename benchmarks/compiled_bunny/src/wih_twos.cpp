#include <embree4/rtcore.h>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef WIH_HAS_OPENMP
#include <omp.h>
#endif

namespace {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 &a, double s) { return {a.x * s, a.y * s, a.z * s}; }

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3 &a) { return std::sqrt(dot(a, a)); }

struct Triangle {
  std::uint32_t a = 0;
  std::uint32_t b = 0;
  std::uint32_t c = 0;
};

struct EmbreeVertex {
  float x;
  float y;
  float z;
  float w;
};

struct EmbreeTriangle {
  std::uint32_t a;
  std::uint32_t b;
  std::uint32_t c;
};

struct Options {
  std::string mode = "twos";
  std::string source_mode = "affine";
  std::string mesh_path;
  std::string points_path;
  std::string output_path;
  std::string summary_path;
  std::string trace_path;
  std::string cavities_path;
  std::array<int, 3> grid{54, 64, 48};
  int walks = 16;
  int repeats = 3;
  int max_steps = 512;
  int threads = 0;
  int trace_sites = 0;
  int mg_cycles = 60;
  int mg_pre_smooth = 3;
  int mg_post_smooth = 3;
  int mg_coarse_smooth = 80;
  std::uint64_t seed = 20260912ULL;
  double epsilon = 5.0e-4;
  double source_y_slope = 0.65;
  double validation_radius_cap = 0.1;
  double validation_value_cap = 0.1;
  double cavity_radius = 0.0;
  double diff_offset = 0.02;
  double mg_tolerance = 1.0e-8;
  double mg_omega = 0.72;
};

[[noreturn]] void usage(const std::string &message = {}) {
  if (!message.empty()) {
    std::cerr << "error: " << message << "\n\n";
  }
  std::cerr
      << "Usage: wih_twos --mesh MESH.ply --points POINTS.csv --output FIELD.csv\n"
      << "                 --summary SUMMARY.json [options]\n\n"
      << "Options:\n"
      << "  --mode NAME           twos, diffwos, voxel, mg, or validate (default twos)\n"
      << "  --grid NX,NY,NZ       voxel resolution (default 54,64,48)\n"
      << "  --validation-radius-cap X  contained-ball radius cap (validate)\n"
      << "  --validation-value-cap X   downward clipping level (validate)\n"
      << "  --walks N             paths per evaluation site (default 16)\n"
      << "  --repeats N           timed repetitions; median is reported (default 3)\n"
      << "  --threads N           OpenMP threads; 0 uses runtime default\n"
      << "  --epsilon X           WoS termination shell (default 5e-4)\n"
      << "  --max-steps N         maximum spheres per path (default 512)\n"
      << "  --seed N              deterministic base seed\n"
      << "  --source-mode NAME    affine or housing (default affine)\n"
      << "  --source-y-slope X    f=1+slope*(y+0.5) (default 0.65)\n"
      << "  --cavities FILE       optional CSV of spherical cavity centers\n"
      << "  --cavity-radius X     common cavity radius (default 0)\n"
      << "  --diff-offset X       inward normal offset for Differential WoS (default 0.02)\n"
      << "  --mg-cycles N         maximum multigrid V-cycles (default 60)\n"
      << "  --mg-pre-smooth N     weighted-Jacobi pre-sweeps (default 3)\n"
      << "  --mg-post-smooth N    weighted-Jacobi post-sweeps (default 3)\n"
      << "  --mg-coarse-smooth N  coarsest-level sweeps (default 80)\n"
      << "  --mg-tolerance X      relative residual tolerance (default 1e-8)\n"
      << "  --mg-omega X          Jacobi relaxation in (0,1) (default 0.72)\n"
      << "  --trace-output FILE   optional CSV of representative paths\n"
      << "  --trace-sites N       number of sites to trace (default 0)\n";
  std::exit(2);
}

Options parse_options(int argc, char **argv) {
  Options options;
  std::unordered_map<std::string, std::string *> strings = {
      {"--mode", &options.mode},
      {"--source-mode", &options.source_mode},
      {"--mesh", &options.mesh_path},
      {"--points", &options.points_path},
      {"--output", &options.output_path},
      {"--summary", &options.summary_path},
      {"--trace-output", &options.trace_path},
      {"--cavities", &options.cavities_path},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      usage();
    }
    if (i + 1 >= argc) {
      usage("missing value for " + key);
    }
    const std::string value = argv[++i];
    if (const auto it = strings.find(key); it != strings.end()) {
      *it->second = value;
    } else if (key == "--walks") {
      options.walks = std::stoi(value);
    } else if (key == "--validation-radius-cap") {
      options.validation_radius_cap = std::stod(value);
    } else if (key == "--validation-value-cap") {
      options.validation_value_cap = std::stod(value);
    } else if (key == "--repeats") {
      options.repeats = std::stoi(value);
    } else if (key == "--max-steps") {
      options.max_steps = std::stoi(value);
    } else if (key == "--threads") {
      options.threads = std::stoi(value);
    } else if (key == "--trace-sites") {
      options.trace_sites = std::stoi(value);
    } else if (key == "--mg-cycles") {
      options.mg_cycles = std::stoi(value);
    } else if (key == "--mg-pre-smooth") {
      options.mg_pre_smooth = std::stoi(value);
    } else if (key == "--mg-post-smooth") {
      options.mg_post_smooth = std::stoi(value);
    } else if (key == "--mg-coarse-smooth") {
      options.mg_coarse_smooth = std::stoi(value);
    } else if (key == "--seed") {
      options.seed = std::stoull(value);
    } else if (key == "--epsilon") {
      options.epsilon = std::stod(value);
    } else if (key == "--source-y-slope") {
      options.source_y_slope = std::stod(value);
    } else if (key == "--cavity-radius") {
      options.cavity_radius = std::stod(value);
    } else if (key == "--diff-offset") {
      options.diff_offset = std::stod(value);
    } else if (key == "--mg-tolerance") {
      options.mg_tolerance = std::stod(value);
    } else if (key == "--mg-omega") {
      options.mg_omega = std::stod(value);
    } else if (key == "--grid") {
      std::string parsed = value;
      std::replace(parsed.begin(), parsed.end(), ',', ' ');
      std::istringstream values(parsed);
      if (!(values >> options.grid[0] >> options.grid[1] >> options.grid[2])) {
        usage("--grid must have the form NX,NY,NZ");
      }
    } else {
      usage("unknown option " + key);
    }
  }
  if (options.mesh_path.empty() || options.points_path.empty() ||
      options.output_path.empty() || options.summary_path.empty()) {
    usage("--mesh, --points, --output, and --summary are required");
  }
  if ((options.mode != "twos" && options.mode != "diffwos" &&
       options.mode != "voxel" && options.mode != "mg" && options.mode != "validate") ||
      (options.source_mode != "affine" && options.source_mode != "housing") ||
      options.walks < 2 || options.repeats < 1 || options.max_steps < 1 ||
      options.epsilon <= 0.0 || options.trace_sites < 0 ||
      options.validation_radius_cap <= 0 || options.validation_value_cap <= 0 ||
      options.mg_cycles < 1 || options.mg_pre_smooth < 1 ||
      options.mg_post_smooth < 1 || options.mg_coarse_smooth < 1 ||
      options.diff_offset <= options.epsilon ||
      options.mg_tolerance <= 0.0 || options.mg_omega <= 0.0 ||
      options.mg_omega >= 1.0 ||
      options.cavity_radius < 0.0 ||
      (!options.cavities_path.empty() && options.cavity_radius <= 0.0)) {
    usage("invalid numeric option");
  }
  if (options.mode == "diffwos" && options.cavities_path.empty()) {
    usage("Differential WoS requires --cavities");
  }
  for (int extent : options.grid) {
    if (extent < 3) usage("every --grid extent must be at least three");
  }
  if (options.mode == "mg") {
    for (int extent : options.grid) {
      if (extent % 2 == 0) {
        usage("geometric multigrid requires odd --grid extents");
      }
    }
  }
  return options;
}

void embree_error(void *, RTCError code, const char *message) {
  std::cerr << "Embree error " << static_cast<int>(code) << ": "
            << (message ? message : "unknown") << '\n';
}

Vec3 closest_point_triangle(const Vec3 &p, const Vec3 &a, const Vec3 &b,
                            const Vec3 &c) {
  const Vec3 ab = b - a;
  const Vec3 ac = c - a;
  const Vec3 ap = p - a;
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;

  const Vec3 bp = p - b;
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return b;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return a + ab * v;
  }

  const Vec3 cp = p - c;
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return c;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return a + ac * w;
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const Vec3 bc = c - b;
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + bc * w;
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return a + ab * v + ac * w;
}

class EmbreeMesh {
 public:
  struct ClosestPointResult {
    double distance = std::numeric_limits<double>::infinity();
    Vec3 point;
  };

  explicit EmbreeMesh(const std::string &path) {
    load_ascii_ply(path);
    const auto start = std::chrono::steady_clock::now();
    device_ = rtcNewDevice(nullptr);
    if (!device_) throw std::runtime_error("rtcNewDevice failed");
    rtcSetDeviceErrorFunction(device_, embree_error, nullptr);
    scene_ = rtcNewScene(device_);
    RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    auto *vertices = static_cast<EmbreeVertex *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        sizeof(EmbreeVertex), vertices_.size()));
    for (std::size_t i = 0; i < vertices_.size(); ++i) {
      vertices[i] = {static_cast<float>(vertices_[i].x),
                     static_cast<float>(vertices_[i].y),
                     static_cast<float>(vertices_[i].z), 0.0F};
    }
    auto *indices = static_cast<EmbreeTriangle *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        sizeof(EmbreeTriangle), triangles_.size()));
    for (std::size_t i = 0; i < triangles_.size(); ++i) {
      indices[i] = {triangles_[i].a, triangles_[i].b, triangles_[i].c};
    }
    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene_, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene_);
    build_seconds_ = seconds_since(start);
  }

  EmbreeMesh(const EmbreeMesh &) = delete;
  EmbreeMesh &operator=(const EmbreeMesh &) = delete;

  ~EmbreeMesh() {
    if (scene_) rtcReleaseScene(scene_);
    if (device_) rtcReleaseDevice(device_);
  }

  double distance(const Vec3 &point) const {
    return closest_point(point).distance;
  }

  ClosestPointResult closest_point(const Vec3 &point) const {
#ifdef WIH_QUERY_COUNTER
    note_query(0);
#endif
    RTCPointQuery query{};
    query.x = static_cast<float>(point.x);
    query.y = static_cast<float>(point.y);
    query.z = static_cast<float>(point.z);
    query.time = 0.0F;
    query.radius = 4.0F;
    RTCPointQueryContext context;
    rtcInitPointQueryContext(&context);
    QueryData data{this, point, {0.0, 0.0, 0.0},
                   std::numeric_limits<double>::infinity()};
    rtcPointQuery(scene_, &query, &context, point_query_callback,
                  &data);
    if (!std::isfinite(data.distance)) {
      throw std::runtime_error("closest-point query failed");
    }
    return {data.distance, data.closest};
  }

  bool inside(const Vec3 &point) const {
    Vec3 direction{1.0, 0.3713906763541037, 0.1570796326794897};
    direction = direction * (1.0 / norm(direction));
    Vec3 origin = point;
    unsigned intersections = 0;
    for (; intersections < 128; ++intersections) {
      RTCRayHit rayhit{};
      rayhit.ray.org_x = static_cast<float>(origin.x);
      rayhit.ray.org_y = static_cast<float>(origin.y);
      rayhit.ray.org_z = static_cast<float>(origin.z);
      rayhit.ray.dir_x = static_cast<float>(direction.x);
      rayhit.ray.dir_y = static_cast<float>(direction.y);
      rayhit.ray.dir_z = static_cast<float>(direction.z);
      rayhit.ray.tnear = 1.0e-6F;
      rayhit.ray.tfar = 4.0F;
      rayhit.ray.time = 0.0F;
      rayhit.ray.mask = 0xffffffffU;
      rayhit.ray.flags = 0U;
      rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
      rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
      RTCIntersectArguments arguments;
      rtcInitIntersectArguments(&arguments);
#ifdef WIH_QUERY_COUNTER
      note_query(1);
#endif
      rtcIntersect1(scene_, &rayhit, &arguments);
      if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) break;
      origin = origin + direction * (static_cast<double>(rayhit.ray.tfar) + 2.0e-6);
    }
    if (intersections == 128) {
      throw std::runtime_error("ray-parity traversal exceeded 128 intersections");
    }
    return intersections % 2U == 1U;
  }

  const std::vector<Vec3> &vertices() const { return vertices_; }
  const std::vector<Triangle> &triangles() const { return triangles_; }

  std::size_t vertex_count() const { return vertices_.size(); }
  std::size_t face_count() const { return triangles_.size(); }
  double build_seconds() const { return build_seconds_; }
  Vec3 bounds_min() const { return bounds_min_; }
  Vec3 bounds_max() const { return bounds_max_; }
  std::size_t input_bytes() const {
    return vertices_.size() * sizeof(Vec3) + triangles_.size() * sizeof(Triangle);
  }

 private:
  struct QueryData {
    const EmbreeMesh *mesh;
    Vec3 point;
    Vec3 closest;
    double distance;
  };

  static bool point_query_callback(RTCPointQueryFunctionArguments *args) {
    auto *data = static_cast<QueryData *>(args->userPtr);
    const Triangle &triangle = data->mesh->triangles_[args->primID];
    const Vec3 closest = closest_point_triangle(
        data->point, data->mesh->vertices_[triangle.a],
        data->mesh->vertices_[triangle.b], data->mesh->vertices_[triangle.c]);
    const double distance = norm(data->point - closest);
    if (distance < static_cast<double>(args->query->radius)) {
      args->query->radius = static_cast<float>(distance);
      data->closest = closest;
      data->distance = distance;
      return true;
    }
    return false;
  }

  static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
  }

  void load_ascii_ply(const std::string &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open mesh: " + path);
    std::string line;
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    bool ascii = false;
    bool header_done = false;
    while (std::getline(input, line)) {
      std::istringstream words(line);
      std::string first;
      words >> first;
      if (first == "format") {
        std::string format;
        words >> format;
        ascii = format == "ascii";
      } else if (first == "element") {
        std::string type;
        std::size_t count = 0;
        words >> type >> count;
        if (type == "vertex") vertex_count = count;
        if (type == "face") face_count = count;
      } else if (first == "end_header") {
        header_done = true;
        break;
      }
    }
    if (!ascii || !header_done || vertex_count == 0 || face_count == 0) {
      throw std::runtime_error("expected a nonempty ASCII triangle PLY mesh");
    }
    vertices_.resize(vertex_count);
    for (Vec3 &vertex : vertices_) {
      if (!(input >> vertex.x >> vertex.y >> vertex.z)) {
        throw std::runtime_error("truncated PLY vertex section");
      }
      bounds_min_.x = std::min(bounds_min_.x, vertex.x);
      bounds_min_.y = std::min(bounds_min_.y, vertex.y);
      bounds_min_.z = std::min(bounds_min_.z, vertex.z);
      bounds_max_.x = std::max(bounds_max_.x, vertex.x);
      bounds_max_.y = std::max(bounds_max_.y, vertex.y);
      bounds_max_.z = std::max(bounds_max_.z, vertex.z);
      std::getline(input, line);
    }
    triangles_.resize(face_count);
    for (Triangle &triangle : triangles_) {
      unsigned count = 0;
      std::uint64_t a = 0, b = 0, c = 0;
      if (!(input >> count >> a >> b >> c) || count != 3 ||
          a >= vertex_count || b >= vertex_count || c >= vertex_count) {
        throw std::runtime_error("expected valid triangular PLY faces");
      }
      triangle = {static_cast<std::uint32_t>(a),
                  static_cast<std::uint32_t>(b),
                  static_cast<std::uint32_t>(c)};
      std::getline(input, line);
    }
  }

  RTCDevice device_ = nullptr;
  RTCScene scene_ = nullptr;
  std::vector<Vec3> vertices_;
  std::vector<Triangle> triangles_;
  Vec3 bounds_min_{std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()};
  Vec3 bounds_max_{-std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()};
  double build_seconds_ = 0.0;
};

std::vector<Vec3> read_points(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open points: " + path);
  std::vector<Vec3> points;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream values(line);
    Vec3 point;
    if (values >> point.x >> point.y >> point.z) points.push_back(point);
  }
  if (points.empty()) throw std::runtime_error("points file contains no points");
  return points;
}

std::vector<Vec3> read_optional_points(const std::string &path) {
  if (path.empty()) return {};
  return read_points(path);
}

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

Vec3 random_unit_vector(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double z = 1.0 - 2.0 * uniform(rng);
  const double phi = 2.0 * std::acos(-1.0) * uniform(rng);
  const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
  return {radial * std::cos(phi), radial * std::sin(phi), z};
}

double square(double value) { return value * value; }

double source(const Vec3 &point, const Options &options) {
#ifdef WIH_CUSTOM_SOURCE
  return custom_source(point);
#else
  if (options.source_mode == "affine") {
    return 1.0 + options.source_y_slope * (point.y + 0.5);
  }
  const double chip_a = std::exp(
      -square((point.x + 0.48) / 0.24)
      -square((point.y + 0.18) / 0.22)
      -square((point.z + 0.08) / 0.16));
  const double chip_b = std::exp(
      -square((point.x - 0.43) / 0.21)
      -square((point.y - 0.19) / 0.20)
      -square((point.z + 0.07) / 0.15));
  const double regulator = std::exp(
      -square((point.x - 0.02) / 0.30)
      -square((point.y + 0.31) / 0.16)
      -square((point.z + 0.02) / 0.19));
  return 0.08 + 5.2 * chip_a + 4.6 * chip_b + 2.0 * regulator;
#endif
}

double beta_22(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  while (true) {
    const double value = uniform(rng);
    if (uniform(rng) <= 4.0 * value * (1.0 - value)) return value;
  }
}

bool inside_cavity(const Vec3 &point, const std::vector<Vec3> &cavities,
                   double cavity_radius) {
  for (const Vec3 &center : cavities) {
    if (norm(point - center) <= cavity_radius) return true;
  }
  return false;
}

double domain_distance(const EmbreeMesh &mesh, const Vec3 &point,
                       const std::vector<Vec3> &cavities,
                       double cavity_radius) {
  double distance = mesh.distance(point);
  for (const Vec3 &center : cavities) {
    distance = std::min(distance, norm(point - center) - cavity_radius);
  }
  return distance;
}

struct BoundaryHit {
  double distance = std::numeric_limits<double>::infinity();
  Vec3 point;
  Vec3 outward_normal;
  int cavity_index = -1;
};

BoundaryHit closest_boundary(const EmbreeMesh &mesh, const Vec3 &point,
                             const std::vector<Vec3> &cavities,
                             double cavity_radius) {
  const EmbreeMesh::ClosestPointResult mesh_hit = mesh.closest_point(point);
  BoundaryHit hit;
  hit.distance = mesh_hit.distance;
  hit.point = mesh_hit.point;
  if (mesh_hit.distance > 0.0) {
    hit.outward_normal = (mesh_hit.point - point) * (1.0 / mesh_hit.distance);
  }
  for (std::size_t index = 0; index < cavities.size(); ++index) {
    const Vec3 radial = point - cavities[index];
    const double radial_length = norm(radial);
    const double distance = radial_length - cavity_radius;
    if (distance < hit.distance && radial_length > 0.0) {
      const Vec3 direction = radial * (1.0 / radial_length);
      hit.distance = distance;
      hit.point = cavities[index] + direction * cavity_radius;
      // The domain excludes each sphere, so its outward normal points into it.
      hit.outward_normal = direction * -1.0;
      hit.cavity_index = static_cast<int>(index);
    }
  }
  return hit;
}

struct WalkResult {
  double value = 0.0;
  int steps = 0;
  bool hit_limit = false;
  std::vector<std::pair<Vec3, double>> trace;
};

WalkResult walk(const EmbreeMesh &mesh, const std::vector<Vec3> &cavities,
                Vec3 point, const Options &options, std::uint64_t seed,
                bool keep_trace) {
  std::mt19937_64 rng(splitmix64(seed));
  WalkResult result;
  if (keep_trace) {
    result.trace.emplace_back(
        point, domain_distance(mesh, point, cavities, options.cavity_radius));
  }
  for (int step = 0; step < options.max_steps; ++step) {
    const double radius = domain_distance(
        mesh, point, cavities, options.cavity_radius);
    if (!(radius > options.epsilon)) return result;
    if (options.source_mode == "affine") {
      result.value += source(point, options) * radius * radius / 6.0;
    } else {
      const Vec3 source_point = point + random_unit_vector(rng) *
          (radius * beta_22(rng));
      result.value += source(source_point, options) * radius * radius / 6.0;
    }
    point = point + random_unit_vector(rng) * radius;
    ++result.steps;
    if (keep_trace) {
      result.trace.emplace_back(
          point, domain_distance(mesh, point, cavities, options.cavity_radius));
    }
  }
  result.hit_limit = true;
  return result;
}

struct SiteResult {
  double mean = 0.0;
  double score = 0.0;
  double score_se = 0.0;
  std::uint64_t steps = 0;
  std::uint64_t limits = 0;
};

SiteResult evaluate_site(const EmbreeMesh &mesh,
                         const std::vector<Vec3> &cavities, const Vec3 &point,
                         std::size_t site_index, int repeat,
                         const Options &options) {
  std::vector<double> values(static_cast<std::size_t>(options.walks));
  SiteResult result;
  for (int sample = 0; sample < options.walks; ++sample) {
    const std::uint64_t key =
        splitmix64(options.seed + 0x2d358dccaa6c78a5ULL) ^
        splitmix64(static_cast<std::uint64_t>(repeat) + 0x8bb84b93962eacc9ULL) ^
        splitmix64(static_cast<std::uint64_t>(site_index) + 0x4b33a62ed433d4a3ULL) ^
        splitmix64(static_cast<std::uint64_t>(sample) + 0x4d5a2da51de1aa47ULL);
    const WalkResult path = walk(mesh, cavities, point, options, key, false);
    values[static_cast<std::size_t>(sample)] = path.value;
    result.steps += static_cast<std::uint64_t>(path.steps);
    result.limits += path.hit_limit ? 1ULL : 0ULL;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  double sum2 = 0.0;
  for (double value : values) sum2 += value * value;
  const double n = static_cast<double>(options.walks);
  result.mean = sum / n;
  result.score = (sum * sum - sum2) / (n * (n - 1.0));

  if (options.walks >= 4) {
    std::vector<double> leave(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      const double leave_sum = sum - values[i];
      const double leave_sum2 = sum2 - values[i] * values[i];
      leave[i] = (leave_sum * leave_sum - leave_sum2) /
                 ((n - 1.0) * (n - 2.0));
    }
    const double leave_mean =
        std::accumulate(leave.begin(), leave.end(), 0.0) / n;
    double spread = 0.0;
    for (double value : leave) {
      spread += (value - leave_mean) * (value - leave_mean);
    }
    result.score_se = std::sqrt((n - 1.0) / n * spread);
  } else {
    result.score_se = std::numeric_limits<double>::quiet_NaN();
  }
  return result;
}

struct BoundaryWalkResult {
  BoundaryHit hit;
  int steps = 0;
  bool hit_limit = false;
};

BoundaryWalkResult walk_to_boundary(const EmbreeMesh &mesh,
                                    const std::vector<Vec3> &cavities,
                                    Vec3 point, const Options &options,
                                    std::uint64_t seed) {
  std::mt19937_64 rng(splitmix64(seed));
  BoundaryWalkResult result;
  for (int step = 0; step < options.max_steps; ++step) {
    result.hit = closest_boundary(
        mesh, point, cavities, options.cavity_radius);
    if (!(result.hit.distance > options.epsilon)) return result;
    point = point + random_unit_vector(rng) * result.hit.distance;
    ++result.steps;
  }
  result.hit_limit = true;
  return result;
}

struct CavityDerivative {
  Vec3 center;
  Vec3 center_se;
  double radius = 0.0;
  double radius_se = 0.0;
};

struct DifferentialSiteResult {
  std::vector<CavityDerivative> derivatives;
  std::uint64_t outer_steps = 0;
  std::uint64_t nested_steps = 0;
  std::uint64_t cavity_terminations = 0;
  std::uint64_t near_seam_terminations = 0;
  std::uint64_t segment_unsafe_terminations = 0;
  std::uint64_t unsafe_offset_terminations = 0;
  std::uint64_t limits = 0;
};

struct OffsetSafety {
  bool near_seam = false;
  bool segment_unsafe = false;
};

OffsetSafety classify_offset_safety(
    const EmbreeMesh &mesh, const std::vector<Vec3> &cavities,
    int hit_cavity, const Vec3 &boundary_point, const Vec3 &offset_point,
    const Options &options) {
  OffsetSafety safety;
  const double radius = options.cavity_radius;
  const double seam_width = options.diff_offset;
  const Vec3 segment = offset_point - boundary_point;
  const double segment_length_squared = dot(segment, segment);
  const double collision_tolerance =
      1.0e-10 * std::max(1.0, std::abs(radius));
  const double collision_radius_squared =
      square(radius + collision_tolerance);

  const Vec3 &hit_center =
      cavities[static_cast<std::size_t>(hit_cavity)];
  for (std::size_t index = 0; index < cavities.size(); ++index) {
    if (static_cast<int>(index) == hit_cavity) continue;

    const Vec3 center_delta = cavities[index] - hit_center;
    const double center_distance = norm(center_delta);
    if (center_distance > collision_tolerance &&
        center_distance < 2.0 * radius) {
      // For equal-radius spheres, the seam circle lies in the perpendicular
      // bisector plane.  This is the exact Euclidean distance from the hit
      // point to that circle; the operational safe strip has width ell.
      const Vec3 axis = center_delta * (1.0 / center_distance);
      const Vec3 seam_center =
          (hit_center + cavities[index]) * 0.5;
      const double seam_radius = std::sqrt(std::max(
          0.0, square(radius) - 0.25 * square(center_distance)));
      const Vec3 relative = boundary_point - seam_center;
      const double axial = dot(relative, axis);
      const Vec3 radial = relative - axis * axial;
      const double seam_distance = std::sqrt(
          square(axial) + square(norm(radial) - seam_radius));
      safety.near_seam = safety.near_seam || seam_distance <= seam_width;
    }

    if (segment_length_squared > 0.0) {
      const double parameter = std::clamp(
          dot(cavities[index] - boundary_point, segment) /
              segment_length_squared,
          0.0, 1.0);
      const Vec3 closest = boundary_point + segment * parameter;
      if (dot(closest - cavities[index], closest - cavities[index]) <=
          collision_radius_squared) {
        safety.segment_unsafe = true;
      }
    }
  }

  // The benchmark housing is well separated from the cavity channel, but this
  // endpoint test keeps the diagnostic meaningful for arbitrary input meshes.
  bool offset_inside = mesh.inside(offset_point);
#ifdef WIH_ENCLOSURE_FALLBACK
  // A parity ray can be degenerate at a triangle edge. For a strictly
  // contained spherical cavity, a larger enclosing ball supplies an
  // independent geometric membership test for this particular offset.
  if (!offset_inside && mesh.inside(hit_center) &&
      mesh.distance(hit_center) > radius + options.diff_offset + options.epsilon &&
      norm(offset_point-hit_center) <= radius + options.diff_offset + options.epsilon*0.5) {
    offset_inside = true;
    note_query(2);
  }
#endif
  if (!offset_inside ||
      !(domain_distance(mesh, offset_point, cavities, radius) >
        options.epsilon)) {
    safety.segment_unsafe = true;
  }
  return safety;
}

DifferentialSiteResult evaluate_differential_site(
    const EmbreeMesh &mesh, const std::vector<Vec3> &cavities,
    const Vec3 &point, std::size_t site_index, int repeat,
    const Options &options) {
  const std::size_t parameter_count = cavities.size();
  std::vector<Vec3> sums(parameter_count);
  std::vector<Vec3> sums_squared(parameter_count);
  std::vector<double> radius_sums(parameter_count, 0.0);
  std::vector<double> radius_sums_squared(parameter_count, 0.0);
  DifferentialSiteResult result;
  result.derivatives.resize(parameter_count);

  for (int sample = 0; sample < options.walks; ++sample) {
    const std::uint64_t outer_key =
        splitmix64(options.seed + 0x6a09e667f3bcc909ULL) ^
        splitmix64(static_cast<std::uint64_t>(repeat) + 0xbb67ae8584caa73bULL) ^
        splitmix64(static_cast<std::uint64_t>(site_index) + 0x3c6ef372fe94f82bULL) ^
        splitmix64(static_cast<std::uint64_t>(sample) + 0xa54ff53a5f1d36f1ULL);
    const BoundaryWalkResult outer = walk_to_boundary(
        mesh, cavities, point, options, outer_key);
    result.outer_steps += static_cast<std::uint64_t>(outer.steps);
    if (outer.hit_limit) {
      ++result.limits;
      continue;
    }
    if (outer.hit.cavity_index < 0) continue;

    ++result.cavity_terminations;
    const Vec3 offset_point = outer.hit.point -
        outer.hit.outward_normal * options.diff_offset;
    const OffsetSafety safety = classify_offset_safety(
        mesh, cavities, outer.hit.cavity_index, outer.hit.point,
        offset_point, options);
    result.near_seam_terminations += safety.near_seam ? 1ULL : 0ULL;
    result.segment_unsafe_terminations +=
        safety.segment_unsafe ? 1ULL : 0ULL;
    if (safety.near_seam || safety.segment_unsafe) {
      ++result.unsafe_offset_terminations;
      continue;
    }
    const std::uint64_t nested_key = splitmix64(
        outer_key ^ 0x510e527fade682d1ULL);
    const WalkResult nested = walk(
        mesh, cavities, offset_point, options, nested_key, false);
    result.nested_steps += static_cast<std::uint64_t>(nested.steps);
    result.limits += nested.hit_limit ? 1ULL : 0ULL;

    const std::size_t cavity =
        static_cast<std::size_t>(outer.hit.cavity_index);
    // For zero Dirichlet data, DiffWoS terminates with
    // dot(u) = V_n * u(x - ell*n) / ell.  A center translation has
    // V_n = n dot dot(c), while a radius change has V_n = -dot(R).
    const double scale = nested.value / options.diff_offset;
    const Vec3 center_sample = outer.hit.outward_normal * scale;
    const double radius_sample = -scale;
    sums[cavity] = sums[cavity] + center_sample;
    sums_squared[cavity].x += center_sample.x * center_sample.x;
    sums_squared[cavity].y += center_sample.y * center_sample.y;
    sums_squared[cavity].z += center_sample.z * center_sample.z;
    radius_sums[cavity] += radius_sample;
    radius_sums_squared[cavity] += radius_sample * radius_sample;
  }

  const double sample_count = static_cast<double>(options.walks);
  for (std::size_t cavity = 0; cavity < parameter_count; ++cavity) {
    CavityDerivative &derivative = result.derivatives[cavity];
    derivative.center = sums[cavity] * (1.0 / sample_count);
    derivative.radius = radius_sums[cavity] / sample_count;
    if (options.walks > 1) {
      const double denominator = sample_count * (sample_count - 1.0);
      derivative.center_se.x = std::sqrt(std::max(
          0.0, (sums_squared[cavity].x -
                sums[cavity].x * sums[cavity].x / sample_count) /
                   denominator));
      derivative.center_se.y = std::sqrt(std::max(
          0.0, (sums_squared[cavity].y -
                sums[cavity].y * sums[cavity].y / sample_count) /
                   denominator));
      derivative.center_se.z = std::sqrt(std::max(
          0.0, (sums_squared[cavity].z -
                sums[cavity].z * sums[cavity].z / sample_count) /
                   denominator));
      derivative.radius_se = std::sqrt(std::max(
          0.0, (radius_sums_squared[cavity] -
                radius_sums[cavity] * radius_sums[cavity] / sample_count) /
                   denominator));
    }
  }
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

double peak_rss_megabytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
      double kilobytes = 0.0;
      std::string unit;
      status >> kilobytes >> unit;
      return kilobytes / 1024.0;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

std::size_t grid_index(int i, int j, int k, const std::array<int, 3> &shape) {
  return (static_cast<std::size_t>(i) * static_cast<std::size_t>(shape[1]) +
          static_cast<std::size_t>(j)) * static_cast<std::size_t>(shape[2]) +
         static_cast<std::size_t>(k);
}

int run_voxel(const EmbreeMesh &mesh, const std::vector<Vec3> &cavities,
              const std::vector<Vec3> &points, const Options &options) {
  using Clock = std::chrono::steady_clock;
  using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Triplet = Eigen::Triplet<double>;
  const std::array<int, 3> shape = options.grid;
  const std::size_t cell_count = static_cast<std::size_t>(shape[0]) *
      static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2]);
  const Vec3 low = mesh.bounds_min();
  const Vec3 high = mesh.bounds_max();
  const std::array<double, 3> spacing{
      (high.x - low.x) / static_cast<double>(shape[0] - 1),
      (high.y - low.y) / static_cast<double>(shape[1] - 1),
      (high.z - low.z) / static_cast<double>(shape[2] - 1)};
  const std::array<double, 3> inverse_h2{
      1.0 / (spacing[0] * spacing[0]),
      1.0 / (spacing[1] * spacing[1]),
      1.0 / (spacing[2] * spacing[2])};

  std::vector<unsigned char> active(cell_count, 0U);
  auto start = Clock::now();
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t linear = 0; linear < static_cast<std::int64_t>(cell_count);
       ++linear) {
    const std::size_t index = static_cast<std::size_t>(linear);
    const int k = static_cast<int>(index % static_cast<std::size_t>(shape[2]));
    const std::size_t remainder = index / static_cast<std::size_t>(shape[2]);
    const int j = static_cast<int>(remainder % static_cast<std::size_t>(shape[1]));
    const int i = static_cast<int>(remainder / static_cast<std::size_t>(shape[1]));
    const Vec3 point{low.x + i * spacing[0], low.y + j * spacing[1],
                     low.z + k * spacing[2]};
    active[index] = (mesh.inside(point) &&
                     !inside_cavity(point, cavities, options.cavity_radius))
                        ? 1U
                        : 0U;
  }
  const double classification_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  start = Clock::now();
  std::vector<int> node_id(cell_count, -1);
  int active_count = 0;
  for (std::size_t index = 0; index < cell_count; ++index) {
    if (active[index]) node_id[index] = active_count++;
  }
  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(active_count) * 7U);
  Eigen::VectorXd rhs(active_count);
  const double diagonal = 2.0 *
      (inverse_h2[0] + inverse_h2[1] + inverse_h2[2]);
  const int delta[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                           {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        const std::size_t linear = grid_index(i, j, k, shape);
        const int row = node_id[linear];
        if (row < 0) continue;
        triplets.emplace_back(row, row, diagonal);
        const Vec3 point{low.x + i * spacing[0], low.y + j * spacing[1],
                         low.z + k * spacing[2]};
        rhs[row] = source(point, options);
        for (int neighbor = 0; neighbor < 6; ++neighbor) {
          const int ni = i + delta[neighbor][0];
          const int nj = j + delta[neighbor][1];
          const int nk = k + delta[neighbor][2];
          if (ni < 0 || nj < 0 || nk < 0 || ni >= shape[0] ||
              nj >= shape[1] || nk >= shape[2]) continue;
          const int column = node_id[grid_index(ni, nj, nk, shape)];
          if (column >= 0) {
            const int dimension = neighbor / 2;
            triplets.emplace_back(row, column, -inverse_h2[dimension]);
          }
        }
      }
    }
  }
  SparseMatrix matrix(active_count, active_count);
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  matrix.makeCompressed();
  triplets.clear();
  triplets.shrink_to_fit();
  const double assembly_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  start = Clock::now();
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                           Eigen::DiagonalPreconditioner<double>> solver;
  solver.setTolerance(1.0e-9);
  solver.setMaxIterations(8000);
  solver.compute(matrix);
  const Eigen::VectorXd solution = solver.solve(rhs);
  const double solve_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("C++ voxel conjugate gradient failed");
  }
  std::vector<double> field(cell_count, 0.0);
  for (std::size_t index = 0; index < cell_count; ++index) {
    if (node_id[index] >= 0) field[index] = solution[node_id[index]];
  }

  start = Clock::now();
  std::vector<double> scores(points.size(), 0.0);
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t site = 0; site < static_cast<std::int64_t>(points.size());
       ++site) {
    const Vec3 &point = points[static_cast<std::size_t>(site)];
    const double coordinate[3] = {
        (point.x - low.x) / spacing[0],
        (point.y - low.y) / spacing[1],
        (point.z - low.z) / spacing[2]};
    int base[3];
    double fraction[3];
    for (int dimension = 0; dimension < 3; ++dimension) {
      const double clipped = std::clamp(
          coordinate[dimension], 0.0, static_cast<double>(shape[dimension] - 1));
      base[dimension] = std::min(
          static_cast<int>(std::floor(clipped)), shape[dimension] - 2);
      fraction[dimension] = clipped - static_cast<double>(base[dimension]);
    }
    double value = 0.0;
    for (int di = 0; di < 2; ++di) {
      for (int dj = 0; dj < 2; ++dj) {
        for (int dk = 0; dk < 2; ++dk) {
          const double weight = (di ? fraction[0] : 1.0 - fraction[0]) *
                                (dj ? fraction[1] : 1.0 - fraction[1]) *
                                (dk ? fraction[2] : 1.0 - fraction[2]);
          value += weight * field[grid_index(
              base[0] + di, base[1] + dj, base[2] + dk, shape)];
        }
      }
    }
    scores[static_cast<std::size_t>(site)] = value * value;
  }
  const double interpolation_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  std::ofstream field_output(options.output_path);
  if (!field_output) throw std::runtime_error("cannot write voxel field output");
  field_output << "site,score\n" << std::setprecision(17);
  for (std::size_t site = 0; site < scores.size(); ++site) {
    field_output << site << ',' << scores[site] << '\n';
  }
  const std::size_t sparse_bytes =
      static_cast<std::size_t>(matrix.nonZeros()) * (sizeof(double) + sizeof(int)) +
      static_cast<std::size_t>(matrix.outerSize() + 1) * sizeof(int);
  const std::size_t accounted_bytes = sparse_bytes + active.size() +
      node_id.size() * sizeof(int) + field.size() * sizeof(double) +
      static_cast<std::size_t>(rhs.size() + solution.size()) * sizeof(double);
  const double total_seconds = mesh.build_seconds() + classification_seconds +
      assembly_seconds + solve_seconds + interpolation_seconds;
  std::ofstream summary(options.summary_path);
  if (!summary) throw std::runtime_error("cannot write voxel summary");
  summary << std::setprecision(17)
          << "{\n"
          << "  \"engine\": \"C++ voxel FD/CG + Embree parity\",\n"
          << "  \"vertices\": " << mesh.vertex_count() << ",\n"
          << "  \"faces\": " << mesh.face_count() << ",\n"
          << "  \"source_mode\": \"" << options.source_mode << "\",\n"
          << "  \"cavities\": " << cavities.size() << ",\n"
          << "  \"cavity_radius\": " << options.cavity_radius << ",\n"
          << "  \"shape\": [" << shape[0] << ", " << shape[1] << ", "
          << shape[2] << "],\n"
          << "  \"grid_cells\": " << cell_count << ",\n"
          << "  \"active_nodes\": " << active_count << ",\n"
          << "  \"threads\": "
#ifdef WIH_HAS_OPENMP
          << omp_get_max_threads()
#else
          << 1
#endif
          << ",\n"
          << "  \"mesh_build_seconds\": " << mesh.build_seconds() << ",\n"
          << "  \"classification_seconds\": " << classification_seconds << ",\n"
          << "  \"assembly_seconds\": " << assembly_seconds << ",\n"
          << "  \"solve_seconds\": " << solve_seconds << ",\n"
          << "  \"interpolation_seconds\": " << interpolation_seconds << ",\n"
          << "  \"total_seconds\": " << total_seconds << ",\n"
          << "  \"cg_iterations\": " << solver.iterations() << ",\n"
          << "  \"cg_reported_error\": " << solver.error() << ",\n"
          << "  \"matrix_nonzeros\": " << matrix.nonZeros() << ",\n"
          << "  \"matrix_megabytes\": "
          << static_cast<double>(sparse_bytes) / (1024.0 * 1024.0) << ",\n"
          << "  \"accounted_working_set_megabytes\": "
          << static_cast<double>(accounted_bytes) / (1024.0 * 1024.0) << ",\n"
          << "  \"peak_rss_megabytes\": " << peak_rss_megabytes() << ",\n"
          << "  \"max_spacing\": "
          << *std::max_element(spacing.begin(), spacing.end()) << "\n}\n";
  std::cout << "C++ voxel: " << shape[0] << 'x' << shape[1] << 'x' << shape[2]
            << ", " << active_count << " active nodes, " << total_seconds
            << " s, " << solver.iterations() << " CG iterations\n";
  return 0;
}

struct MGLevel {
  std::array<int, 3> shape{0, 0, 0};
  Vec3 low;
  std::array<double, 3> spacing{0.0, 0.0, 0.0};
  std::array<double, 3> inverse_h2{0.0, 0.0, 0.0};
  double diagonal = 0.0;
  std::size_t active_count = 0;
  std::vector<unsigned char> active;
  std::vector<double> u;
  std::vector<double> rhs;
  std::vector<double> residual;
  std::vector<double> scratch;
};

MGLevel make_mg_level(const EmbreeMesh &mesh,
                      const std::vector<Vec3> &cavities,
                      double cavity_radius,
                      const std::array<int, 3> &shape) {
  MGLevel level;
  level.shape = shape;
  level.low = mesh.bounds_min();
  const Vec3 high = mesh.bounds_max();
  level.spacing = {
      (high.x - level.low.x) / static_cast<double>(shape[0] - 1),
      (high.y - level.low.y) / static_cast<double>(shape[1] - 1),
      (high.z - level.low.z) / static_cast<double>(shape[2] - 1)};
  level.inverse_h2 = {
      1.0 / (level.spacing[0] * level.spacing[0]),
      1.0 / (level.spacing[1] * level.spacing[1]),
      1.0 / (level.spacing[2] * level.spacing[2])};
  level.diagonal = 2.0 * (level.inverse_h2[0] + level.inverse_h2[1] +
                          level.inverse_h2[2]);
  const std::size_t count = static_cast<std::size_t>(shape[0]) *
      static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2]);
  level.active.assign(count, 0U);
  level.u.assign(count, 0.0);
  level.rhs.assign(count, 0.0);
  level.residual.assign(count, 0.0);
  level.scratch.assign(count, 0.0);

#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t linear = 0; linear < static_cast<std::int64_t>(count);
       ++linear) {
    const std::size_t index = static_cast<std::size_t>(linear);
    const int k = static_cast<int>(index % static_cast<std::size_t>(shape[2]));
    const std::size_t remainder = index / static_cast<std::size_t>(shape[2]);
    const int j = static_cast<int>(remainder % static_cast<std::size_t>(shape[1]));
    const int i = static_cast<int>(remainder / static_cast<std::size_t>(shape[1]));
    const Vec3 point{level.low.x + i * level.spacing[0],
                     level.low.y + j * level.spacing[1],
                     level.low.z + k * level.spacing[2]};
    const bool is_active = mesh.inside(point) &&
        !inside_cavity(point, cavities, cavity_radius);
    level.active[index] = is_active ? 1U : 0U;
  }
  level.active_count = static_cast<std::size_t>(std::count(
      level.active.begin(), level.active.end(), static_cast<unsigned char>(1U)));
  return level;
}

double mg_apply_vector_at(const MGLevel &level,
                          const std::vector<double> &values,
                          int i, int j, int k) {
  const std::size_t center = grid_index(i, j, k, level.shape);
  double value = level.diagonal * values[center];
  if (i > 0) {
    value -= level.inverse_h2[0] *
        values[grid_index(i - 1, j, k, level.shape)];
  }
  if (i + 1 < level.shape[0]) {
    value -= level.inverse_h2[0] *
        values[grid_index(i + 1, j, k, level.shape)];
  }
  if (j > 0) {
    value -= level.inverse_h2[1] *
        values[grid_index(i, j - 1, k, level.shape)];
  }
  if (j + 1 < level.shape[1]) {
    value -= level.inverse_h2[1] *
        values[grid_index(i, j + 1, k, level.shape)];
  }
  if (k > 0) {
    value -= level.inverse_h2[2] *
        values[grid_index(i, j, k - 1, level.shape)];
  }
  if (k + 1 < level.shape[2]) {
    value -= level.inverse_h2[2] *
        values[grid_index(i, j, k + 1, level.shape)];
  }
  return value;
}

double mg_apply_at(const MGLevel &level, int i, int j, int k) {
  return mg_apply_vector_at(level, level.u, i, j, k);
}

void mg_smooth(MGLevel &level, int sweeps, double omega) {
  for (int sweep = 0; sweep < sweeps; ++sweep) {
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int i = 0; i < level.shape[0]; ++i) {
      for (int j = 0; j < level.shape[1]; ++j) {
        for (int k = 0; k < level.shape[2]; ++k) {
          const std::size_t index = grid_index(i, j, k, level.shape);
          if (!level.active[index]) {
            level.scratch[index] = 0.0;
            continue;
          }
          const double correction =
              (level.rhs[index] - mg_apply_at(level, i, j, k)) /
              level.diagonal;
          level.scratch[index] = level.u[index] + omega * correction;
        }
      }
    }
    level.u.swap(level.scratch);
  }
}

double mg_residual_norm(MGLevel &level) {
  double squared_norm = 0.0;
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for collapse(3) reduction(+ : squared_norm) schedule(static)
#endif
  for (int i = 0; i < level.shape[0]; ++i) {
    for (int j = 0; j < level.shape[1]; ++j) {
      for (int k = 0; k < level.shape[2]; ++k) {
        const std::size_t index = grid_index(i, j, k, level.shape);
        if (!level.active[index]) {
          level.residual[index] = 0.0;
          continue;
        }
        const double value =
            level.rhs[index] - mg_apply_at(level, i, j, k);
        level.residual[index] = value;
        squared_norm += value * value;
      }
    }
  }
  return std::sqrt(squared_norm);
}

void mg_restrict(const MGLevel &fine, MGLevel &coarse) {
  std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
  for (int i = 0; i < coarse.shape[0]; ++i) {
    for (int j = 0; j < coarse.shape[1]; ++j) {
      for (int k = 0; k < coarse.shape[2]; ++k) {
        const std::size_t coarse_index = grid_index(i, j, k, coarse.shape);
        if (!coarse.active[coarse_index]) continue;
        double weighted = 0.0;
        for (int di = -1; di <= 1; ++di) {
          const int fi = 2 * i + di;
          if (fi < 0 || fi >= fine.shape[0]) continue;
          const double wi = di == 0 ? 2.0 : 1.0;
          for (int dj = -1; dj <= 1; ++dj) {
            const int fj = 2 * j + dj;
            if (fj < 0 || fj >= fine.shape[1]) continue;
            const double wj = dj == 0 ? 2.0 : 1.0;
            for (int dk = -1; dk <= 1; ++dk) {
              const int fk = 2 * k + dk;
              if (fk < 0 || fk >= fine.shape[2]) continue;
              const double wk = dk == 0 ? 2.0 : 1.0;
              weighted += wi * wj * wk *
                  fine.residual[grid_index(fi, fj, fk, fine.shape)];
            }
          }
        }
        coarse.rhs[coarse_index] = weighted / 64.0;
      }
    }
  }
}

void mg_prolongate_and_add(const MGLevel &coarse, MGLevel &fine) {
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int i = 0; i < fine.shape[0]; ++i) {
    for (int j = 0; j < fine.shape[1]; ++j) {
      for (int k = 0; k < fine.shape[2]; ++k) {
        const std::size_t fine_index = grid_index(i, j, k, fine.shape);
        if (!fine.active[fine_index]) {
          fine.scratch[fine_index] = 0.0;
          continue;
        }
        const int base_i = i / 2;
        const int base_j = j / 2;
        const int base_k = k / 2;
        const double tx = (i % 2) ? 0.5 : 0.0;
        const double ty = (j % 2) ? 0.5 : 0.0;
        const double tz = (k % 2) ? 0.5 : 0.0;
        double correction = 0.0;
        for (int di = 0; di <= (tx > 0.0 ? 1 : 0); ++di) {
          for (int dj = 0; dj <= (ty > 0.0 ? 1 : 0); ++dj) {
            for (int dk = 0; dk <= (tz > 0.0 ? 1 : 0); ++dk) {
              const int ci = std::min(base_i + di, coarse.shape[0] - 1);
              const int cj = std::min(base_j + dj, coarse.shape[1] - 1);
              const int ck = std::min(base_k + dk, coarse.shape[2] - 1);
              const double weight = (di ? tx : 1.0 - tx) *
                                    (dj ? ty : 1.0 - ty) *
                                    (dk ? tz : 1.0 - tz);
              correction += weight *
                  coarse.u[grid_index(ci, cj, ck, coarse.shape)];
            }
          }
        }
        fine.scratch[fine_index] = correction;
      }
    }
  }
  double numerator = 0.0;
  double denominator = 0.0;
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for collapse(3) reduction(+ : numerator, denominator) schedule(static)
#endif
  for (int i = 0; i < fine.shape[0]; ++i) {
    for (int j = 0; j < fine.shape[1]; ++j) {
      for (int k = 0; k < fine.shape[2]; ++k) {
        const std::size_t index = grid_index(i, j, k, fine.shape);
        if (!fine.active[index]) continue;
        const double applied =
            mg_apply_vector_at(fine, fine.scratch, i, j, k);
        numerator += fine.residual[index] * applied;
        denominator += applied * applied;
      }
    }
  }
  const double alpha = denominator > 0.0
      ? std::clamp(numerator / denominator, 0.0, 1.5)
      : 0.0;
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(fine.u.size()); ++index) {
    const std::size_t linear = static_cast<std::size_t>(index);
    fine.u[linear] += alpha * fine.scratch[linear];
  }
}

void mg_v_cycle(std::vector<MGLevel> &levels, std::size_t level_index,
                const Options &options) {
  MGLevel &level = levels[level_index];
  if (level_index + 1 == levels.size()) {
    mg_smooth(level, options.mg_coarse_smooth, options.mg_omega);
    return;
  }
  mg_smooth(level, options.mg_pre_smooth, options.mg_omega);
  mg_residual_norm(level);
  MGLevel &coarse = levels[level_index + 1];
  mg_restrict(level, coarse);
  std::fill(coarse.u.begin(), coarse.u.end(), 0.0);
  mg_v_cycle(levels, level_index + 1, options);
  mg_prolongate_and_add(coarse, level);
  mg_smooth(level, options.mg_post_smooth, options.mg_omega);
}

int run_multigrid(const EmbreeMesh &mesh, const std::vector<Vec3> &cavities,
                  const std::vector<Vec3> &points, const Options &options) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  std::vector<MGLevel> levels;
  std::array<int, 3> shape = options.grid;
  while (true) {
    levels.push_back(make_mg_level(
        mesh, cavities, options.cavity_radius, shape));
    if (*std::min_element(shape.begin(), shape.end()) <= 9) break;
    shape = {(shape[0] + 1) / 2, (shape[1] + 1) / 2,
             (shape[2] + 1) / 2};
  }
  const double hierarchy_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  MGLevel &fine = levels.front();
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (int i = 0; i < fine.shape[0]; ++i) {
    for (int j = 0; j < fine.shape[1]; ++j) {
      for (int k = 0; k < fine.shape[2]; ++k) {
        const std::size_t index = grid_index(i, j, k, fine.shape);
        if (!fine.active[index]) continue;
        const Vec3 point{fine.low.x + i * fine.spacing[0],
                         fine.low.y + j * fine.spacing[1],
                         fine.low.z + k * fine.spacing[2]};
        fine.rhs[index] = source(point, options);
      }
    }
  }
  double rhs_squared_norm = 0.0;
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for reduction(+ : rhs_squared_norm) schedule(static)
#endif
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(fine.rhs.size()); ++index) {
    rhs_squared_norm += fine.rhs[static_cast<std::size_t>(index)] *
                        fine.rhs[static_cast<std::size_t>(index)];
  }
  const double rhs_norm = std::sqrt(rhs_squared_norm);

  start = Clock::now();
  int cycles = 0;
  double relative_residual = rhs_norm > 0.0 ? 1.0 : 0.0;
  for (; cycles < options.mg_cycles &&
         relative_residual > options.mg_tolerance; ++cycles) {
    mg_v_cycle(levels, 0, options);
    relative_residual = mg_residual_norm(fine) / std::max(rhs_norm, 1.0e-300);
  }
  const double solve_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  start = Clock::now();
  std::vector<double> scores(points.size(), 0.0);
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t site = 0; site < static_cast<std::int64_t>(points.size());
       ++site) {
    const Vec3 &point = points[static_cast<std::size_t>(site)];
    const double coordinate[3] = {
        (point.x - fine.low.x) / fine.spacing[0],
        (point.y - fine.low.y) / fine.spacing[1],
        (point.z - fine.low.z) / fine.spacing[2]};
    int base[3];
    double fraction[3];
    for (int dimension = 0; dimension < 3; ++dimension) {
      const double clipped = std::clamp(
          coordinate[dimension], 0.0,
          static_cast<double>(fine.shape[dimension] - 1));
      base[dimension] = std::min(
          static_cast<int>(std::floor(clipped)), fine.shape[dimension] - 2);
      fraction[dimension] = clipped - static_cast<double>(base[dimension]);
    }
    double value = 0.0;
    for (int di = 0; di < 2; ++di) {
      for (int dj = 0; dj < 2; ++dj) {
        for (int dk = 0; dk < 2; ++dk) {
          const double weight = (di ? fraction[0] : 1.0 - fraction[0]) *
                                (dj ? fraction[1] : 1.0 - fraction[1]) *
                                (dk ? fraction[2] : 1.0 - fraction[2]);
          value += weight * fine.u[grid_index(
              base[0] + di, base[1] + dj, base[2] + dk, fine.shape)];
        }
      }
    }
    scores[static_cast<std::size_t>(site)] = value * value;
  }
  const double interpolation_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  std::ofstream field_output(options.output_path);
  if (!field_output) {
    throw std::runtime_error("cannot write multigrid field output");
  }
  field_output << "site,score\n" << std::setprecision(17);
  for (std::size_t site = 0; site < scores.size(); ++site) {
    field_output << site << ',' << scores[site] << '\n';
  }

  std::size_t accounted_bytes = 0;
  for (const MGLevel &level : levels) {
    accounted_bytes += level.active.size() * sizeof(unsigned char);
    accounted_bytes += 4U * level.u.size() * sizeof(double);
  }
  const double total_seconds = mesh.build_seconds() + hierarchy_seconds +
      solve_seconds + interpolation_seconds;
  std::ofstream summary(options.summary_path);
  if (!summary) {
    throw std::runtime_error("cannot write multigrid summary");
  }
  summary << std::setprecision(17)
          << "{\n"
          << "  \"engine\": \"C++ matrix-free geometric multigrid + Embree parity\",\n"
          << "  \"vertices\": " << mesh.vertex_count() << ",\n"
          << "  \"faces\": " << mesh.face_count() << ",\n"
          << "  \"source_mode\": \"" << options.source_mode << "\",\n"
          << "  \"cavities\": " << cavities.size() << ",\n"
          << "  \"cavity_radius\": " << options.cavity_radius << ",\n"
          << "  \"shape\": [" << fine.shape[0] << ", " << fine.shape[1]
          << ", " << fine.shape[2] << "],\n"
          << "  \"levels\": " << levels.size() << ",\n"
          << "  \"grid_cells\": " << fine.active.size() << ",\n"
          << "  \"active_nodes\": " << fine.active_count << ",\n"
          << "  \"threads\": "
#ifdef WIH_HAS_OPENMP
          << omp_get_max_threads()
#else
          << 1
#endif
          << ",\n"
          << "  \"mesh_build_seconds\": " << mesh.build_seconds() << ",\n"
          << "  \"hierarchy_seconds\": " << hierarchy_seconds << ",\n"
          << "  \"solve_seconds\": " << solve_seconds << ",\n"
          << "  \"interpolation_seconds\": " << interpolation_seconds << ",\n"
          << "  \"total_seconds\": " << total_seconds << ",\n"
          << "  \"v_cycles\": " << cycles << ",\n"
          << "  \"relative_residual\": " << relative_residual << ",\n"
          << "  \"tolerance\": " << options.mg_tolerance << ",\n"
          << "  \"pre_smooth\": " << options.mg_pre_smooth << ",\n"
          << "  \"post_smooth\": " << options.mg_post_smooth << ",\n"
          << "  \"coarse_smooth\": " << options.mg_coarse_smooth << ",\n"
          << "  \"omega\": " << options.mg_omega << ",\n"
          << "  \"accounted_working_set_megabytes\": "
          << static_cast<double>(accounted_bytes) / (1024.0 * 1024.0) << ",\n"
          << "  \"peak_rss_megabytes\": " << peak_rss_megabytes() << ",\n"
          << "  \"max_spacing\": "
          << *std::max_element(fine.spacing.begin(), fine.spacing.end())
          << "\n}\n";
  std::cout << "C++ geometric multigrid: " << fine.shape[0] << 'x'
            << fine.shape[1] << 'x' << fine.shape[2] << ", "
            << fine.active_count << " active nodes, " << levels.size()
            << " levels, " << total_seconds << " s, " << cycles
            << " V-cycles, residual " << relative_residual << '\n';
  return 0;
}

void write_differential_field(
    const std::string &path, const std::vector<Vec3> &points,
    const std::vector<DifferentialSiteResult> &results,
    std::size_t cavity_count) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot write differential field output: " + path);
  }
  output << "site,x,y,z";
  for (std::size_t cavity = 0; cavity < cavity_count; ++cavity) {
    output << ",dc" << cavity << "_x,dc" << cavity << "_y,dc" << cavity
           << "_z,dc" << cavity << "_se_x,dc" << cavity << "_se_y,dc"
           << cavity << "_se_z,dR" << cavity << ",dR" << cavity << "_se";
  }
  output << "\n" << std::setprecision(17);
  for (std::size_t site = 0; site < points.size(); ++site) {
    output << site << ',' << points[site].x << ',' << points[site].y << ','
           << points[site].z;
    for (const CavityDerivative &derivative : results[site].derivatives) {
      output << ',' << derivative.center.x << ',' << derivative.center.y << ','
             << derivative.center.z << ',' << derivative.center_se.x << ','
             << derivative.center_se.y << ',' << derivative.center_se.z << ','
             << derivative.radius << ',' << derivative.radius_se;
    }
    output << '\n';
  }
}

int run_differential_wos(const EmbreeMesh &mesh,
                         const std::vector<Vec3> &cavities,
                         const std::vector<Vec3> &points,
                         const Options &options) {
  std::vector<double> times;
  std::vector<DifferentialSiteResult> retained(points.size());
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    std::vector<DifferentialSiteResult> results(points.size());
    const auto start = std::chrono::steady_clock::now();
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(points.size()); ++index) {
      results[static_cast<std::size_t>(index)] = evaluate_differential_site(
          mesh, cavities, points[static_cast<std::size_t>(index)],
          static_cast<std::size_t>(index), repeat, options);
    }
    times.push_back(std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count());
    if (repeat == 0) retained = std::move(results);
  }

  std::uint64_t outer_steps = 0;
  std::uint64_t nested_steps = 0;
  std::uint64_t cavity_terminations = 0;
  std::uint64_t near_seam_terminations = 0;
  std::uint64_t segment_unsafe_terminations = 0;
  std::uint64_t unsafe_offset_terminations = 0;
  std::uint64_t limits = 0;
  for (const DifferentialSiteResult &result : retained) {
    outer_steps += result.outer_steps;
    nested_steps += result.nested_steps;
    cavity_terminations += result.cavity_terminations;
    near_seam_terminations += result.near_seam_terminations;
    segment_unsafe_terminations += result.segment_unsafe_terminations;
    unsafe_offset_terminations += result.unsafe_offset_terminations;
    limits += result.limits;
  }
  if (limits != 0) {
    throw std::runtime_error("one or more differential paths hit --max-steps");
  }
  write_differential_field(
      options.output_path, points, retained, cavities.size());

  const double runtime = median(times);
  const double paths = static_cast<double>(points.size()) * options.walks;
  std::ofstream summary(options.summary_path);
  if (!summary) {
    throw std::runtime_error("cannot write differential summary: " +
                             options.summary_path);
  }
  summary << std::setprecision(17)
          << "{\n"
          << "  \"engine\": \"Embree 4 Differential WoS\",\n"
          << "  \"mode\": \"diffwos\",\n"
          << "  \"vertices\": " << mesh.vertex_count() << ",\n"
          << "  \"faces\": " << mesh.face_count() << ",\n"
          << "  \"source_mode\": \"" << options.source_mode << "\",\n"
          << "  \"cavities\": " << cavities.size() << ",\n"
          << "  \"gradient_parameters\": " << 4 * cavities.size() << ",\n"
          << "  \"cavity_radius\": " << options.cavity_radius << ",\n"
          << "  \"sites\": " << points.size() << ",\n"
          << "  \"walks_per_site\": " << options.walks << ",\n"
          << "  \"seed\": " << options.seed << ",\n"
          << "  \"epsilon\": " << options.epsilon << ",\n"
          << "  \"normal_offset\": " << options.diff_offset << ",\n"
          << "  \"seam_strip_multiplier\": 1,\n"
          << "  \"runtime_seconds_median\": " << runtime << ",\n"
          << "  \"outer_mean_steps\": "
          << static_cast<double>(outer_steps) / paths << ",\n"
          << "  \"nested_mean_steps_per_outer_path\": "
          << static_cast<double>(nested_steps) / paths << ",\n"
          << "  \"cavity_termination_fraction\": "
          << static_cast<double>(cavity_terminations) / paths << ",\n"
          << "  \"near_seam_fraction\": "
          << static_cast<double>(near_seam_terminations) / paths << ",\n"
          << "  \"segment_unsafe_fraction\": "
          << static_cast<double>(segment_unsafe_terminations) / paths << ",\n"
          << "  \"unsafe_offset_fraction\": "
          << static_cast<double>(unsafe_offset_terminations) / paths << ",\n"
          << "  \"unsafe_given_cavity_fraction\": "
          << (cavity_terminations == 0
                  ? 0.0
                  : static_cast<double>(unsafe_offset_terminations) /
                        static_cast<double>(cavity_terminations))
          << ",\n"
          << "  \"near_seam_terminations\": "
          << near_seam_terminations << ",\n"
          << "  \"segment_unsafe_terminations\": "
          << segment_unsafe_terminations << ",\n"
          << "  \"unsafe_offset_terminations\": "
          << unsafe_offset_terminations << ",\n"
          << "  \"mesh_build_seconds\": " << mesh.build_seconds() << ",\n"
          << "  \"peak_rss_megabytes\": " << peak_rss_megabytes() << ",\n"
          << "  \"timings_seconds\": [";
  for (std::size_t index = 0; index < times.size(); ++index) {
    if (index) summary << ", ";
    summary << times[index];
  }
  summary << "]\n}\n";
  std::cout << "Embree Differential WoS: " << points.size() << " sites x "
            << options.walks << " walks, " << 4 * cavities.size()
            << " parameters, median " << runtime << " s\n";
  return 0;
}

void write_field(const std::string &path, const std::vector<Vec3> &points,
                 const std::vector<SiteResult> &results) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write field output: " + path);
  output << "site,x,y,z,u,score,score_se,steps,limit_hits\n";
  output << std::setprecision(17);
  for (std::size_t i = 0; i < points.size(); ++i) {
    output << i << ',' << points[i].x << ',' << points[i].y << ','
           << points[i].z << ',' << results[i].mean << ',' << results[i].score
           << ',' << results[i].score_se << ',' << results[i].steps << ','
           << results[i].limits << '\n';
  }
}

void write_trace(const std::string &path, const EmbreeMesh &mesh,
                 const std::vector<Vec3> &cavities,
                 const std::vector<Vec3> &points, const Options &options) {
  if (path.empty() || options.trace_sites == 0) return;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write trace output: " + path);
  output << "path,step,x,y,z,radius\n" << std::setprecision(17);
  const int count = std::min<int>(options.trace_sites, points.size());
  for (int i = 0; i < count; ++i) {
    const std::size_t site = static_cast<std::size_t>(i) * points.size() /
                             static_cast<std::size_t>(count);
    const WalkResult result = walk(mesh, cavities, points[site], options,
                                   options.seed + 900000ULL + site, true);
    for (std::size_t step = 0; step < result.trace.size(); ++step) {
      const Vec3 &point = result.trace[step].first;
      output << i << ',' << step << ',' << point.x << ',' << point.y << ','
             << point.z << ',' << result.trace[step].second << '\n';
    }
  }
}

void write_summary(const std::string &path, const EmbreeMesh &mesh,
                   const std::vector<Vec3> &cavities, const Options &options,
                   std::size_t site_count,
                   const std::vector<double> &times,
                   std::uint64_t total_steps, std::uint64_t total_limits) {
  const double runtime = median(times);
  const double paths = static_cast<double>(site_count) * options.walks;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write summary: " + path);
  output << std::setprecision(17)
         << "{\n"
         << "  \"engine\": \"Embree 4\",\n"
         << "  \"vertices\": " << mesh.vertex_count() << ",\n"
         << "  \"faces\": " << mesh.face_count() << ",\n"
         << "  \"source_mode\": \"" << options.source_mode << "\",\n"
         << "  \"cavities\": " << cavities.size() << ",\n"
         << "  \"cavity_radius\": " << options.cavity_radius << ",\n"
         << "  \"sites\": " << site_count << ",\n"
         << "  \"walks_per_site\": " << options.walks << ",\n"
         << "  \"repeats\": " << options.repeats << ",\n"
         << "  \"threads\": "
#ifdef WIH_HAS_OPENMP
         << omp_get_max_threads()
#else
         << 1
#endif
         << ",\n"
         << "  \"epsilon\": " << options.epsilon << ",\n"
         << "  \"mesh_build_seconds\": " << mesh.build_seconds() << ",\n"
         << "  \"runtime_seconds_median\": " << runtime << ",\n"
         << "  \"runtime_seconds_min\": "
         << *std::min_element(times.begin(), times.end()) << ",\n"
         << "  \"paths_per_second\": " << paths / runtime << ",\n"
         << "  \"distance_queries_per_second\": "
         << (static_cast<double>(total_steps) + paths) / runtime << ",\n"
         << "  \"mean_steps\": "
         << static_cast<double>(total_steps) / paths << ",\n"
         << "  \"max_step_limit_hits\": " << total_limits << ",\n"
         << "  \"mesh_input_megabytes\": "
         << static_cast<double>(mesh.input_bytes()) / (1024.0 * 1024.0) << ",\n"
         << "  \"peak_rss_megabytes\": " << peak_rss_megabytes() << ",\n"
         << "  \"timings_seconds\": [";
  for (std::size_t i = 0; i < times.size(); ++i) {
    if (i) output << ", ";
    output << times[i];
  }
  output << "]\n}\n";
}

#include "minimum_benefit.hpp"

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
#ifdef WIH_HAS_OPENMP
    if (options.threads > 0) omp_set_num_threads(options.threads);
#endif
    const EmbreeMesh mesh(options.mesh_path);
    const std::vector<Vec3> points = read_points(options.points_path);
    const std::vector<Vec3> cavities = read_optional_points(options.cavities_path);
    if (options.mode == "validate") {
      return run_minimum_benefit(mesh, cavities, points, options);
    }
    if (options.mode == "voxel") {
      return run_voxel(mesh, cavities, points, options);
    }
    if (options.mode == "mg") {
      return run_multigrid(mesh, cavities, points, options);
    }
    if (options.mode == "diffwos") {
      return run_differential_wos(mesh, cavities, points, options);
    }
    std::vector<double> times;
    std::vector<SiteResult> retained(points.size());
    std::uint64_t retained_steps = 0;
    std::uint64_t retained_limits = 0;

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      std::vector<SiteResult> results(points.size());
      const auto start = std::chrono::steady_clock::now();
#ifdef WIH_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(points.size()); ++index) {
        results[static_cast<std::size_t>(index)] = evaluate_site(
            mesh, cavities, points[static_cast<std::size_t>(index)],
            static_cast<std::size_t>(index), repeat, options);
      }
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      times.push_back(elapsed);
      if (repeat == 0) retained = std::move(results);
    }
    for (const SiteResult &result : retained) {
      retained_steps += result.steps;
      retained_limits += result.limits;
    }
    if (retained_limits != 0) {
      throw std::runtime_error("one or more paths hit --max-steps");
    }
    write_field(options.output_path, points, retained);
    write_trace(options.trace_path, mesh, cavities, points, options);
    write_summary(options.summary_path, mesh, cavities, options, points.size(), times,
                  retained_steps, retained_limits);
    std::cout << "Embree TWoS: " << points.size() << " sites x "
              << options.walks << " walks, median " << median(times)
              << " s, " << std::fixed << std::setprecision(0)
              << (static_cast<double>(points.size()) * options.walks /
                  median(times))
              << " paths/s\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "fatal: " << exception.what() << '\n';
    return 1;
  }
}
