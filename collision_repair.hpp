// Isolated translation-only repair. Included after the pinned geometry helpers.
// The official rm_overlap, Clipper precision, and scoring threshold are unchanged.
#pragma once
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace gfpack_repair {
using Point = std::pair<double, double>;
using Poly = std::vector<Point>;
using Layout = std::vector<Poly>;
using Box = std::array<double, 4>;
constexpr double bounds_tol = 1e-9;
// Two cells of the existing default Clipper grid; a geometric clearance, not
// a change to the intersection precision or the caller's area epsilon.
constexpr double clearance = 2.0 / 128.0;
constexpr int evaluation_limit = 100000;
constexpr double time_limit_seconds = 8.0;
using TimePoint = std::chrono::steady_clock::time_point;
struct TimeLimit {};
inline void check_time(const TimePoint& started) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() > time_limit_seconds)
        throw TimeLimit{};
}

inline double dot(Point a, Point b) { return a.first*b.first + a.second*b.second; }

inline Poly placed(const Poly& rotated, Point action) {
    Poly result;
    result.reserve(rotated.size());
    for (auto p : rotated) result.emplace_back(p.first + action.first, p.second + action.second);
    return result;
}

inline Box box(const Poly& p) {
    Box b{{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}};
    for (auto v : p) {
        b[0] = std::min(b[0], v.first); b[1] = std::min(b[1], v.second);
        b[2] = std::max(b[2], v.first); b[3] = std::max(b[3], v.second);
    }
    return b;
}

inline bool inside(const Box& b, double w, double h) {
    return b[0] >= -bounds_tol && b[1] >= -bounds_tol &&
           b[2] <= w + bounds_tol && b[3] <= h + bounds_tol;
}

inline double intersection(const Poly& a, const Box& ab, const Poly& b, const Box& bb) {
    if (ab[2] <= bb[0] || bb[2] <= ab[0] || ab[3] <= bb[1] || bb[3] <= ab[1]) return 0;
    // Reuse the pinned precision=2 Clipper implementation without changing eps.
    return areapoly(a, b);
}

struct Measurement {
    std::vector<double> pairs;
    std::vector<Box> boxes;
    double total = 0, maximum = 0;
    bool in_bounds = true;
};

inline Measurement measure(const Layout& layout, double w, double h, const TimePoint& started) {
    Measurement m;
    const size_t n = layout.size();
    m.pairs.resize(n*n, 0);
    for (const auto& p : layout) {
        check_time(started);
        m.boxes.push_back(box(p));
        m.in_bounds = m.in_bounds && inside(m.boxes.back(), w, h);
    }
    for (size_t i = 0; i < n; ++i) for (size_t j = i+1; j < n; ++j) {
        check_time(started);
        double area = intersection(layout[i], m.boxes[i], layout[j], m.boxes[j]);
        if (!std::isfinite(area) || area < 0) throw std::runtime_error("Nonfinite intersection");
        m.pairs[i*n+j] = area;
        m.total += area;
        m.maximum = std::max(m.maximum, area);
    }
    check_time(started);
    return m;
}

inline std::pair<double,double> projection(const Poly& p, Point u) {
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (auto v : p) { const double x = dot(v,u); lo = std::min(lo,x); hi = std::max(hi,x); }
    return {lo, hi};
}

inline std::pair<double,double> travel(const Box& b, Point u, double w, double h) {
    double lo = -std::numeric_limits<double>::infinity(), hi = -lo;
    for (int axis = 0; axis < 2; ++axis) {
        const double v = axis == 0 ? u.first : u.second;
        if (std::abs(v) < 1e-14) continue;
        const double extent = axis == 0 ? w : h;
        double a = -b[axis] / v, c = (extent - b[axis+2]) / v;
        if (a > c) std::swap(a,c);
        lo = std::max(lo,a); hi = std::min(hi,c);
    }
    return {lo,hi};
}

inline void add_axis(std::vector<Point>& axes, Point u) {
    const double length = std::hypot(u.first,u.second);
    if (length < 1e-12) return;
    u.first /= length; u.second /= length;
    if (u.first < -1e-12 || (std::abs(u.first) <= 1e-12 && u.second < 0)) {
        u.first = -u.first; u.second = -u.second;
    }
    for (auto v : axes) if (std::abs(dot(u,v)) > 1-1e-12) return;
    axes.push_back(u);
}

inline std::vector<Point> axes_for(const Poly& a, const Poly& b, const TimePoint& started) {
    std::vector<Point> axes{{1,0},{0,1}};
    for (const Poly* p : {&a,&b}) for (size_t k = 0; k < p->size(); ++k) {
        check_time(started);
        const auto x = (*p)[k], y = (*p)[(k+1)%p->size()];
        add_axis(axes, {x.second-y.second,y.first-x.first});
    }
    check_time(started);
    return axes;
}

inline py::dict repair(std::vector<int> ids, std::vector<double> rotates,
                      std::vector<Point> actions, Layout polys,
                      double width, double height, double eps=1., int max_iterations=2000) {
    const size_t n = ids.size();
    if (!n || n > 2048 || n != rotates.size() || n != actions.size() ||
        !std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0 ||
        width > 1e9 || height > 1e9 || !std::isfinite(eps) || eps < 0 ||
        max_iterations < 0 || max_iterations > 2000)
        throw std::invalid_argument("Invalid repair dimensions, epsilon or iteration budget");
    const auto original_actions = actions;
    Layout rotated, layout;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= polys.size() || polys[ids[i]].size() < 3 || polys[ids[i]].size() > 4096 ||
            !std::isfinite(rotates[i]) || !std::isfinite(actions[i].first) || !std::isfinite(actions[i].second))
            throw std::invalid_argument("Invalid repair input");
        Poly r;
        for (auto p : polys[ids[i]]) {
            if (!std::isfinite(p.first) || !std::isfinite(p.second) ||
                std::abs(p.first) > 1e9 || std::abs(p.second) > 1e9)
                throw std::invalid_argument("Invalid polygon coordinate");
            r.push_back(rotateP(p,rotates[i]));
        }
        if (calculatePolygonArea(r) <= 0) throw std::invalid_argument("Degenerate polygon");
        rotated.push_back(r);
        layout.push_back(placed(r,actions[i]));
        for (auto p : layout.back()) if (!std::isfinite(p.first) || !std::isfinite(p.second) ||
            std::abs(p.first) > 2e9 || std::abs(p.second) > 2e9)
            throw std::invalid_argument("Placed coordinate exceeds safe geometry range");
    }
    const auto started = std::chrono::steady_clock::now();
    Measurement initial, current;
    bool initial_measured = false;
    int iterations = 0, evaluations = 0;
    std::string status;
    try {
    initial = measure(layout,width,height,started);
    current = initial;
    initial_measured = true;
    if (!initial.in_bounds) status = "infeasible_bounds";
    else if (initial.maximum <= eps) status = "converged";
    while (status.empty()) {
        if (iterations >= max_iterations) { status = "iteration_limit"; break; }
        double best_total = current.total;
        double best_motion = std::numeric_limits<double>::infinity();
        bool found = false, budget_exhausted = false;
        size_t best_i = 0, best_j = 0;
        Point best_a, best_b;
        Poly best_pa, best_pb;
        for (size_t i = 0; i < n && !budget_exhausted; ++i)
        for (size_t j = i+1; j < n && !budget_exhausted; ++j) {
            check_time(started);
            if (current.pairs[i*n+j] <= eps) continue;
            for (auto axis : axes_for(layout[i],layout[j],started)) {
                check_time(started);
                if (budget_exhausted) break;
                for (int sign : {1,-1}) {
                    if (budget_exhausted) break;
                    const Point u{sign*axis.first,sign*axis.second};
                    const auto ap = projection(layout[i],u), bp = projection(layout[j],u);
                    double separation = bp.second - ap.first + clearance;
                    if (separation <= 0) continue;
                    const auto at = travel(current.boxes[i],u,width,height);
                    const auto bt = travel(current.boxes[j],u,width,height);
                    // tA-tB=separation, with both new polygons inside the strip.
                    double lo = std::max(at.first,bt.first+separation);
                    double hi = std::min(at.second,bt.second+separation);
                    // A strip may admit exact contact but have no room for an
                    // extra gap (e.g. two width-10 squares in a width-20 strip).
                    // In that case use the full-vertex contact separation.
                    if (lo > hi) {
                        separation = bp.second - ap.first;
                        lo = std::max(at.first,bt.first+separation);
                        hi = std::min(at.second,bt.second+separation);
                    }
                    if (lo > hi) continue;
                    std::vector<double> splits;
                    // Use the least-motion legal split plus either anchored
                    // piece. Do not translate the pair together to arbitrary
                    // container edges merely to shave off tiny overlap areas.
                    for (double preferred : {separation/2,0.,separation}) {
                        const double t = std::min(hi,std::max(lo,preferred));
                        bool duplicate = false;
                        for (double old : splits) if (std::abs(t-old) < 1e-12) duplicate = true;
                        if (!duplicate) splits.push_back(t);
                    }
                    for (double ta : splits) {
                        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                        if (elapsed > time_limit_seconds || evaluations >= evaluation_limit) {
                            status = elapsed > time_limit_seconds ? "time_limit" : "evaluation_limit";
                            budget_exhausted = true; break;
                        }
                        ++evaluations;
                        const double tb = ta-separation;
                        const Point a{actions[i].first+ta*u.first,actions[i].second+ta*u.second};
                        const Point b{actions[j].first+tb*u.first,actions[j].second+tb*u.second};
                        const Poly pa = placed(rotated[i],a), pb = placed(rotated[j],b);
                        const Box ba = box(pa), bb = box(pb);
                        if (!inside(ba,width,height) || !inside(bb,width,height)) continue;
                        // Unmoved pair intersections stay cached; recompute every pair
                        // touching either moved piece, including all third parties.
                        double total = 0;
                        for (size_t x = 0; x < n; ++x) for (size_t y = x+1; y < n; ++y) {
                            check_time(started);
                            const Poly& px = x==i ? pa : (x==j ? pb : layout[x]);
                            const Poly& py = y==i ? pa : (y==j ? pb : layout[y]);
                            const Box& bx = x==i ? ba : (x==j ? bb : current.boxes[x]);
                            const Box& by = y==i ? ba : (y==j ? bb : current.boxes[y]);
                            const double area = x==i || x==j || y==i || y==j ?
                                intersection(px,bx,py,by) : current.pairs[x*n+y];
                            total += area;
                        }
                        const double motion = ta*ta + tb*tb;
                        const double improve_tol = 1e-12*std::max(1.,current.total);
                        if (std::isfinite(total) && total < current.total-improve_tol &&
                            (!found || total < best_total-improve_tol ||
                             (std::abs(total-best_total) <= improve_tol && motion < best_motion))) {
                            found = true; best_total = total; best_motion = motion;
                            best_i=i; best_j=j; best_a=a; best_b=b; best_pa=pa; best_pb=pb;
                        }
                    }
                }
            }
        }
        if (budget_exhausted) break;
        if (!found) { status = "stalled"; break; }
        actions[best_i]=best_a; actions[best_j]=best_b;
        layout[best_i]=best_pa; layout[best_j]=best_pb;
        const auto updated = measure(layout,width,height,started);
        if (!updated.in_bounds || !(updated.total < current.total)) {
            status = "stalled"; break;
        }
        current = updated;
        ++iterations;
        // The success condition refers to the actual accepted layout after all
        // movements, not to pre-movement collision or boundary flags.
        if (current.maximum <= eps) status = "converged";
    }
    } catch (const TimeLimit&) {
        status = "time_limit";
    }
    const bool success = status == "converged";
    const auto returned = success ? current : initial;
    if (!success) actions = original_actions;
    py::dict diagnostics;
    diagnostics["initial_max_pairwise_area"] = initial.maximum;
    diagnostics["initial_sum_pairwise_area"] = initial.total;
    diagnostics["best_attempt_max_pairwise_area"] = current.maximum;
    diagnostics["best_attempt_sum_pairwise_area"] = current.total;
    diagnostics["returned_max_pairwise_area"] = returned.maximum;
    diagnostics["returned_sum_pairwise_area"] = returned.total;
    diagnostics["returned_within_bounds"] = returned.in_bounds;
    diagnostics["initial_geometry_measured"] = initial_measured;
    if (!initial_measured) {
        // No partial measurement may be mistaken for zero overlap.
        for (const char* key : {"initial_max_pairwise_area", "initial_sum_pairwise_area",
             "best_attempt_max_pairwise_area", "best_attempt_sum_pairwise_area",
             "returned_max_pairwise_area", "returned_sum_pairwise_area", "returned_within_bounds"})
            diagnostics[key] = py::none();
    }
    diagnostics["candidate_evaluations"] = evaluations;
    diagnostics["clipper_decimal_precision"] = 2;
    diagnostics["separation_clearance"] = clearance;
    diagnostics["evaluation_limit"] = evaluation_limit;
    diagnostics["time_limit_seconds"] = time_limit_seconds;
    diagnostics["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    py::dict result;
    result["actions_xy"] = actions; result["status"] = status;
    result["iterations"] = iterations; result["rolled_back"] = !success;
    result["diagnostics"] = diagnostics;
    return result;
}
}  // namespace gfpack_repair
