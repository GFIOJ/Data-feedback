import unittest
import tempfile
from pathlib import Path
import numpy as np
from paper1.common import Struct, bounds
from paper1.rfid_module import defaults, read_prob, batch_read_prob
from paper1.plot_environment import plot_environment
from paper1.plot_environment_multiscale import plot_environment_multiscale
from paper1.game_hover_solver import solve_game, coverage_probability
from paper1.legacy_planner import LegacyRandom
from paper1.uav_mission_planner import uav_mission_planner, first_path_conflict
from paper1.main_stage1_hover_planner import main_stage1_hover_planner
from paper1.paper2_export import _first_read_on_segment, export_paper2_inputs
from paper1.paper2_dataset import (Paper2Dataset, recompute_data_amount,
                                   recompute_data_amounts, _read_events)


class PythonTests(unittest.TestCase):
    def test_stacked_boxes_merge_for_collision_checks(self):
        boxes=[Struct(x=0,y=0,z=0,width=2,length=3,height=2),
               Struct(x=0,y=0,z=2,width=2,length=3,height=2),
               Struct(x=0,y=0,z=5,width=2,length=3,height=1)]
        low,high=bounds(boxes)
        self.assertEqual(len(low),2)
        np.testing.assert_array_equal(high[:,2]-low[:,2],[4,1])

    def test_legacy_randperm_fixture(self):
        np.testing.assert_array_equal(LegacyRandom(1).permutation(10)+1,[3,6,5,7,4,8,9,1,10,2])

    def test_default_scene_counts(self):
        c,t,s=plot_environment(Visualize=False)
        self.assertEqual(len(c),1043)
        self.assertEqual(int((t.normal[:,1]<0).sum()),531)
        self.assertEqual((c[0].length,c[0].width,c[0].height),(12.192,2.438,2.896))
        np.testing.assert_array_equal(np.unique(t.z,return_counts=True)[1],[288,288,275,192])
        expected=(150-(4*(6*(2.438+.4))+3*15))/2
        self.assertEqual(c[0].x,expected)

    def test_multiscale_counts_and_sampling(self):
        for index in range(1,6):
            c,t,_,_,preset=plot_environment_multiscale(SceneIndex=index,NumTags=17,Visualize=False)
            self.assertEqual(len(c),index*500)
            self.assertEqual(len(t.x),17)
            self.assertEqual(len(set(t.id)),17)

    def test_aggregate_yard_represents_physical_inventory(self):
        c,t,s=plot_environment(AggregateYard=True,SceneWidth=2000,SceneLength=2000,
            NumBlocksX=10,NumBlocksY=10,StripsPerRegion=4,GroundSlots=60000,
            StackTiers=5,BoundaryMargin=50,NumTags=8000,Visualize=False)
        self.assertEqual((len(c),len(t.x)),(400,8000))
        self.assertEqual((s.num_regions,s.aggregate_stack_count,s.physical_container_count),(100,400,300000))
        self.assertAlmostEqual(s.ground_utilization,60000*12.192*2.438/4_000_000)
        self.assertGreaterEqual(min(block.x for block in c),50)
        self.assertLessEqual(max(block.x+block.width for block in c),1950)
        self.assertGreaterEqual(min(block.y for block in c),50)
        self.assertLessEqual(max(block.y+block.length for block in c),1950)

    def test_rfid_uses_only_radius_and_reader_angle(self):
        r=defaults()
        candidates=np.array([[0,0,0,1,0,0]],float)
        tags=np.array([[8,0,0],[4,4*np.sqrt(3),0],[0,8,0],[8.01,0,0]],float)
        p,stats=batch_read_prob(candidates,tags,r)
        np.testing.assert_array_equal(p.toarray(),[[1,1,0,0]])
        self.assertEqual(stats.readable,2)
        for j,expected in enumerate([1,1,0,0]):
            self.assertEqual(read_prob(candidates[0,:3],candidates[0,3:],tags[j],r),expected)

    def test_coverage_formula_and_unique_players(self):
        p=np.array([[.9,0],[0,.9],[.6,.6]])
        c=np.array([[1,1,1,0,1,0],[2,1,1,0,1,0],[3,1,1,0,1,0]])
        _,ids,rate,history=solve_game(p,c,dict(target_coverage=.8,max_players=3,n_restarts=3,verbose=False))
        self.assertEqual(len(ids),len(set(ids)))
        self.assertAlmostEqual(rate,coverage_probability(p,ids).mean())
        self.assertGreaterEqual(rate,.8)
        for trace in history.convergence:
            self.assertTrue(np.all(np.diff(trace)>=-1e-10))

    def test_default_target_requires_full_coverage(self):
        p=np.eye(2)
        c=np.array([[0,0,0,1,0,0],[1,0,0,1,0,0]],float)
        _,ids,rate,_=solve_game(p,c,dict(max_players=2,n_restarts=2,verbose=False))
        self.assertEqual(len(ids),2)
        self.assertEqual(rate,1.0)

    def test_full_scene_uses_local_flyby_scans(self):
        result=main_stage1_hover_planner(Visualize=False,Verbose=False)
        self.assertEqual(result.Khover,48)
        self.assertEqual(result.num_hover_points,0)
        self.assertEqual(result.coverageRate,1.0)
        self.assertEqual(result.history.method,'exact_set_cover')

    def test_static_mode_has_certified_minimum_cover(self):
        result=main_stage1_hover_planner(Visualize=False,Verbose=False,EnableFlyByRFID=False)
        self.assertEqual(result.Khover,96)
        self.assertEqual(result.coverageRate,1.0)
        self.assertTrue(result.history.optimal)

    def test_continuous_crossing(self):
        a=np.array([[-1,0,1,0],[1,0,1,1]])
        b=np.array([[0,-1,1,0],[0,1,1,1]])
        self.assertIsNotNone(first_path_conflict(a,b,.1))
        b[:,3]+=2
        self.assertIsNone(first_path_conflict(a,b,.1))

    def test_both_planner_modes(self):
        scene=Struct(x_range=np.array([0,20]),y_range=np.array([0,20]),z_range=np.array([0,10]))
        hp=np.array([[5,5,3,0,1,0],[10,10,3,0,-1,0],[15,15,3,0,1,0]])
        for mode in (True,False):
            sol,ms,stats=uav_mission_planner(hp,[],scene,defaults(),2,DepotPos=[1,1,3],
                GridRes=2,InflateRadius=0,UAVSpeed=5,SafetyRadius=.5,GameRestarts=1,
                GameMaxIter=3,MaxImproveIters=1,TopKExactRefine=2,
                EnableConflictFeedback=False,LegacyCompatibility=mode,Verbose=False)
            self.assertTrue(sol.validation.passed,sol.validation.reasons)
            self.assertAlmostEqual(ms,max(sol.uavCosts))
            self.assertAlmostEqual(stats.alignedObjective,ms+.05*sol.soc)
            np.testing.assert_array_equal(sorted(np.concatenate(sol.assignment)),np.arange(3))

    def test_moving_scan_service_is_executed(self):
        scene=Struct(x_range=np.array([0,20]),y_range=np.array([0,20]),z_range=np.array([0,10]))
        hp=np.array([[5,5,3,0,1,0]],float)
        loop=np.array([[5,5,3],[5,7,3],[5,5,3]],float)
        sol,_,_=uav_mission_planner(hp,[],scene,defaults(),1,DepotPos=[1,1,3],ServicePaths=[loop],
            GridRes=2,InflateRadius=0,UAVSpeed=2,SafetyRadius=0,GameRestarts=1,GameMaxIter=2,
            MaxImproveIters=1,TopKExactRefine=1,EnableConflictFeedback=False,
            LegacyCompatibility=False,Verbose=False)
        self.assertTrue(sol.validation.passed,sol.validation.reasons)
        self.assertTrue(any(np.allclose(point[:3],[5,7,3]) for point in sol.paths[0]))

    def test_continuous_scan_first_read_time(self):
        segment=np.array([0,0,0,10,0,0,0,1,0],float)
        time=_first_read_on_segment(np.array([9,4,0]),segment,2.,1.,8.,np.deg2rad(60))
        self.assertAlmostEqual(time,2+9-np.sqrt(48))

    def test_paper2_export_time_state_queue_and_amount_recompute(self):
        container=Struct(x=1.,y=1.,z=0.,width=2.,length=3.,height=5.,aggregate=True)
        tags=Struct(id=np.array([42]),x=np.array([5.]),y=np.array([0.]),z=np.array([1.]),
                    owner_idx=np.array([0]),normal=np.array([[0.,1.,0.]]))
        stage1=Struct(flyByRFID=True,tagPos=np.array([[5.,0.,1.]]),Tags=tags,
            Containers=[container],scene=Struct(x_range=np.array([0.,20.]),y_range=np.array([0.,20.]),
                z_range=np.array([0.,10.]),physical_container_count=5,
                physical_container_dimensions=Struct(length=12.192,width=2.438,height=2.896)),
            rfid=Struct(maxReadRange=8.,maxScanAngleDeg=120.),
            scanSegments=np.array([[0.,-4.,1.,10.,-4.,1.,0.,1.,0.]]),
            coverageRate=1.,Khover=1,num_hover_points=0)
        path=np.array([[0.,-4.,1.,.25],[0.,-4.,1.,1.],[10.,-4.,1.,3.],[0.,-4.,1.,5.]])
        timeline=Struct(serviceWindows=np.array([[1.,1.,3.]]))
        validation=Struct(passed=True,reasons=[])
        stage2=Struct(solution=Struct(paths=[path],timelinePerUAV=[timeline],soc=4.75),
            numUAVs=1,uavSpeed=5.,makespan=5.,loadImbalance=0.,validation=validation,
            depotPos=np.array([0.,-4.,1.]))
        with tempfile.TemporaryDirectory() as folder:
            export_paper2_inputs(stage1,stage2,folder,slot_seconds=1.,
                data_per_collection_bytes=10,base_station_position_m=[2.,2.,9.],
                scenario_config={'seed':7})
            data=Paper2Dataset(folder)
            self.assertFalse(data.query_uav(0,.1)['communication_online'])
            self.assertTrue(np.isnan(data.query_uav(0,.1)['position_m']).all())
            self.assertTrue(data.query_uav(0,.25)['communication_online'])
            self.assertFalse(data.query_uav(0,5.)['communication_online'])
            with np.load(Path(folder)/'slot_arrivals.npz') as arrivals:
                self.assertEqual(arrivals['first_read_slot_zero_based'][0],1)
                self.assertEqual(arrivals['first_read_target_ids'][0],42)
                self.assertEqual(arrivals['new_data_bytes_total'][1],10)
                self.assertEqual(arrivals['sendable_new_data_bytes_per_uav'][0,2],10)
            self.assertTrue(data.node_states['boundary_slot'][0,0])
            unchanged_files = ['trajectories.npz', 'node_states.npz', 'environment.npz']
            before_files = {name: (Path(folder)/name).read_bytes() for name in unchanged_files}
            before_events = _read_events(Path(folder)/'collection_events.csv')
            recompute_data_amount(folder,data_per_collection_bytes=2_000_000)
            after_events = _read_events(Path(folder)/'collection_events.csv')
            for before, after in zip(before_events, after_events):
                for key in set(before) - {'data_bytes', 'data_bits'}:
                    self.assertEqual(before[key], after[key])
            self.assertEqual(before_files,
                {name: (Path(folder)/name).read_bytes() for name in unchanged_files})
            with np.load(Path(folder)/'slot_arrivals.npz') as arrivals:
                self.assertEqual(arrivals['new_data_bytes_total'].sum(),2_000_000)
                self.assertEqual(arrivals['new_data_bits_total'].sum(),16_000_000)
            scenario = recompute_data_amounts(folder, np.array([625_001], dtype=np.int64),
                                              metadata={'data_size_seed': 2027})
            self.assertNotIn('data_per_collection_bytes', scenario)
            self.assertEqual(scenario['data_size_mode'], 'per_data_unit')
            self.assertEqual(scenario['data_size_seed'], 2027)
            event = _read_events(Path(folder)/'collection_events.csv')[0]
            self.assertEqual((event['data_bytes'], event['data_bits']), (625_001, 5_000_008))
            with np.load(Path(folder)/'slot_arrivals.npz') as arrivals:
                self.assertEqual(arrivals['new_data_bytes_total'].sum(), 625_001)


if __name__=='__main__':
    unittest.main()
