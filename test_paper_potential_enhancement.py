"""Check the numerical energy surrogate, not equivalence with author code."""
import unittest
import numpy as np
from paper_sat_geometry import PaperSATGeometry
from paper_enhancement import enhance,field


class PotentialTests(unittest.TestCase):
    def energy(self,polys,a,height=10.):
        return field(PaperSATGeometry(height),polys,list(range(len(polys))),a,height)[0]

    def test_finite_difference_gradient_matches_area_weighted_field(self):
        convex=[[[0,0],[2,0],[2,1],[0,1]],[[0,0],[1,0],[1,1],[0,1]]]
        concave=[[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]],[[.9,1.5],[1.9,1.5],[1.9,2.5],[.9,2.5]]]
        for polys,coords in ((convex,[[0.,1.],[1.6,1.2]]),(concave,[[0.,2.],[0.,2.]])):
            a=np.array([[[x,y,1.,0.] for x,y in coords]])
            r=self.energy(polys,a)
            from shapely.geometry import Polygon
            area=np.array([Polygon(p).area for p in polys]);mass=area/area.sum()
            expected=-mass[:,None]*np.array(r['repulsion'])[0]
            for i in range(2):
                for axis in range(2):
                    p=a.copy();m=a.copy();p[0,i,axis]+=1e-6;m[0,i,axis]-=1e-6
                    measured=(self.energy(polys,p)['collision_potential'][0]-self.energy(polys,m)['collision_potential'][0])/2e-6
                    self.assertAlmostEqual(measured,expected[i,axis],places=7)

    def test_boundary_energy_gradient(self):
        polys=[[[0,0],[1,0],[1,1],[0,1]]]
        for y in (-.2,9.3):
            a=np.array([[[0.,y,1.,0.]]]);r=self.energy(polys,a)
            p=a.copy();m=a.copy();p[0,0,1]+=1e-6;m[0,0,1]-=1e-6
            derivative=(self.energy(polys,p)['collision_potential'][0]-self.energy(polys,m)['collision_potential'][0])/2e-6
            self.assertAlmostEqual(derivative,-r['repulsion'][0][0][1],places=7)

    def test_backtracking_decreases_potential_and_exposes_stationary_infeasibility(self):
        square=[[0,0],[1,0],[1,1],[0,1]]
        a=np.array([[[0.,0.,1.,0.],[0.,0.,1.,0.]]])
        out,post=enhance(PaperSATGeometry(1.),[square],[0,0],a,100,1.,
            policy='paper_potential_descent',trace_candidates=True)
        first=post['trace'][0]
        self.assertEqual(first['accepted_collision_step'],[.5])
        self.assertLess(first['potential_after_collision'][0],first['potential_before'][0])
        self.assertEqual(post['statuses'],['stationary_infeasible'])
        r=field(PaperSATGeometry(1.),[square],[0,0],out,1.)[0]
        self.assertGreater(r['max_pairwise_area'][0],1e-9)
        self.assertTrue(np.array_equal(out[:,:,2:],a[:,:,2:]))

    def test_disabled_budget_preserves_every_coordinate(self):
        a=np.array([[[2.,1.,1.,0.]]]);square=[[0,0],[1,0],[1,1],[0,1]]
        out,post=enhance(PaperSATGeometry(),[square],[0],a,0,policy='paper_potential_descent')
        np.testing.assert_array_equal(out,a);self.assertEqual(post['iterations_executed'],0)


if __name__=='__main__':unittest.main()
