// Eq. 9 translation repulsion and the gap primitive for a documented
// reconstruction of GFPack++ inference. Requires collision_repair.hpp helpers.
#pragma once

namespace gfpack_paper {
using Point = gfpack_repair::Point;
using Poly = gfpack_repair::Poly;
using Layout = gfpack_repair::Layout;
using Box = gfpack_repair::Box;

inline bool convex(const Poly& p) {
    int sign=0;
    for (size_t k=0;k<p.size();++k) {
        auto a=p[k], b=p[(k+1)%p.size()], c=p[(k+2)%p.size()];
        double z=(b.first-a.first)*(c.second-b.second)-(b.second-a.second)*(c.first-b.first);
        if (std::abs(z)<1e-10) continue;
        int s=z>0 ? 1 : -1;
        if (sign && sign!=s) return false;
        sign=s;
    }
    return true;
}

inline Point sat_displacement(const Poly& a,const Poly& b) {
    // Standard full-projection SAT MTV for convex polygons. For a concave
    // contour this is a conservative edge-axis proxy, not an exact global MTV.
    std::vector<Point> axes{{1.,0.},{0.,1.}};
    for (const Poly* p : {&a,&b}) for (size_t k=0;k<p->size();++k) {
        auto x=(*p)[k], y=(*p)[(k+1)%p->size()];
        gfpack_repair::add_axis(axes,{x.second-y.second,y.first-x.first});
    }
    double best=std::numeric_limits<double>::infinity();
    Point result{0,0};
    for (auto u:axes) {
        const auto ap=gfpack_repair::projection(a,u), bp=gfpack_repair::projection(b,u);
        if (ap.second<=bp.first || bp.second<=ap.first) return {0,0};
        for (double shift : {bp.second-ap.first, bp.first-ap.second}) {
            if (std::abs(shift)<best) {
                best=std::abs(shift); result={u.first*shift,u.second*shift};
            }
        }
    }
    return result;
}

inline py::dict repulsion_all(std::vector<int> ids,
        std::vector<std::vector<double>> rotates,
        std::vector<std::vector<Point>> actions, Layout polys,
        double height=1200., double collision_area_eps=0.) {
    const size_t n=ids.size(), batches=actions.size();
    if (!n || !batches || n>2048 || batches>2048 || rotates.size()!=batches ||
        !std::isfinite(height) || height<=0 || height>1e9 ||
        !std::isfinite(collision_area_eps) || collision_area_eps<0)
        throw std::invalid_argument("Invalid paper-constraint dimensions");
    std::vector<double> areas(n);
    std::vector<bool> is_convex(n);
    for (size_t i=0;i<n;++i) {
        if (ids[i]<0 || (size_t)ids[i]>=polys.size() || polys[ids[i]].size()<3 || polys[ids[i]].size()>4096)
            throw std::invalid_argument("Invalid paper polygon ID or contour");
        for (auto p:polys[ids[i]]) if (!std::isfinite(p.first) || !std::isfinite(p.second) ||
            std::abs(p.first)>1e9 || std::abs(p.second)>1e9)
            throw std::invalid_argument("Invalid paper polygon coordinate");
        areas[i]=calculatePolygonArea(polys[ids[i]]);
        if (areas[i]<=0) throw std::invalid_argument("Degenerate paper polygon");
        is_convex[i]=convex(polys[ids[i]]);
    }
    for (size_t batch=0;batch<batches;++batch) {
        if (actions[batch].size()!=n || rotates[batch].size()!=n)
            throw std::invalid_argument("Paper candidate dimensions disagree");
        for (size_t i=0;i<n;++i) if (!std::isfinite(actions[batch][i].first) ||
            !std::isfinite(actions[batch][i].second) || !std::isfinite(rotates[batch][i]) ||
            std::abs(actions[batch][i].first)>1e9 || std::abs(actions[batch][i].second)>1e9)
            throw std::invalid_argument("Nonfinite or unsafe paper candidate");
    }
    std::vector<std::vector<Point>> fields(batches,std::vector<Point>(n,{0,0}));
    std::vector<double> max_areas(batches,0), sums(batches,0), max_y_violation(batches,0), min_x(batches,0);
    std::vector<int> pair_counts(batches,0), concave_pairs(batches,0);
    std::vector<std::string> errors(batches);
    #pragma omp parallel for
    for (int batch=0;batch<(int)batches;++batch) {
        try {
            Layout placed;
            std::vector<Box> boxes;
            for (size_t i=0;i<n;++i) {
                Poly p;
                for (auto v:polys[ids[i]]) p.push_back(translateP(rotateP(v,rotates[batch][i]),actions[batch][i]));
                placed.push_back(p); boxes.push_back(gfpack_repair::box(p));
                if (i==0 || boxes.back()[0]<min_x[batch]) min_x[batch]=boxes.back()[0];
                // An infinite horizontal strip has only the lower/upper y wall.
                const double below=std::max(0.,-boxes.back()[1]);
                const double above=std::max(0.,boxes.back()[3]-height);
                fields[batch][i].second+=below-above;
                max_y_violation[batch]=std::max(max_y_violation[batch],std::max(below,above));
            }
            for (size_t i=0;i<n;++i) for (size_t j=i+1;j<n;++j) {
                const double area=gfpack_repair::intersection(placed[i],boxes[i],placed[j],boxes[j]);
                max_areas[batch]=std::max(max_areas[batch],area); sums[batch]+=area;
                if (area<=collision_area_eps) continue;
                ++pair_counts[batch];
                if (!is_convex[i] || !is_convex[j]) ++concave_pairs[batch];
                const auto d=sat_displacement(placed[i],placed[j]);
                const double ai=areas[j]/(areas[i]+areas[j]), aj=areas[i]/(areas[i]+areas[j]);
                fields[batch][i].first+=ai*d.first; fields[batch][i].second+=ai*d.second;
                fields[batch][j].first-=aj*d.first; fields[batch][j].second-=aj*d.second;
            }
        } catch (const std::exception& e) { errors[batch]=e.what(); }
    }
    for (size_t batch=0;batch<batches;++batch) {
        if (!errors[batch].empty()) throw std::runtime_error("Constraint candidate "+std::to_string(batch)+": "+errors[batch]);
        for (auto d:fields[batch]) if (!std::isfinite(d.first) || !std::isfinite(d.second))
            throw std::runtime_error("Nonfinite paper repulsion");
    }
    py::dict result;
    result["repulsion"]=fields;
    result["max_pairwise_area"]=max_areas; result["sum_pairwise_area"]=sums;
    result["colliding_pairs"]=pair_counts; result["concave_proxy_pairs"]=concave_pairs;
    result["height_violation"]=max_y_violation;
    result["min_x"]=min_x;
    result["gradient_convention"]="repulsion=-grad(Phi), add g^2*h*repulsion";
    result["mtv_method"]="full-projection convex SAT; conservative edge-axis proxy for concave contours";
    return result;
}

inline std::vector<std::vector<Point>> gap_xy_all(std::vector<int> ids,
        std::vector<std::vector<double>> rotates,
        std::vector<std::vector<Point>> actions, Layout polys) {
    // Validate via the same serial checks before invoking OpenMP helpers.
    repulsion_all(ids,rotates,actions,polys,1200.,0.);
    std::vector<std::string> errors(actions.size());
    #pragma omp parallel for
    for (int batch=0;batch<(int)actions.size();++batch) {
        try {
            workx(ids,rotates[batch],actions[batch],polys);
            worky(ids,rotates[batch],actions[batch],polys);
        } catch (const std::exception& e) { errors[batch]=e.what(); }
    }
    for (const auto& e:errors) if (!e.empty()) throw std::runtime_error(e);
    return actions;
}
} // namespace gfpack_paper
