from ....io.geotable.file import read_files as rf
from .. import _accessors as to_test
from ...shapes import Point, Chain, Polygon, Rectangle, LineSegment
from ....common import pandas, RTOL, ATOL
from ....examples import get_path
import numpy as np
import unittest as ut

PANDAS_EXTINCT = pandas is None


@ut.skipIf(PANDAS_EXTINCT, "Missing pandas")
class Test_Accessors(ut.TestCase):
    def setUp(self):
        self.polygons = rf(get_path("Polygon.shp"))
        self.points = rf(get_path("Point.shp"))
        self.lines = rf(get_path("Line.shp"))

    def test_area(self):

        with self.assertRaises(AttributeError):
            to_test.area(self.points)
        with self.assertRaises(AttributeError):
            to_test.area(self.lines)

        areas = to_test.area(self.polygons).values
        answer = [0.000284, 0.000263, 0.001536]
        np.testing.assert_allclose(answer, areas, rtol=RTOL, atol=ATOL * 10)

    def test_bbox(self):

        with self.assertRaises(AttributeError):
            to_test.bbox(self.points)
        with self.assertRaises(AttributeError):
            to_test.bbox(self.lines)

        answer = [
            [
                -0.010809397704086565,
                -0.26282711761789435,
                0.12787295484449185,
                -0.250785835510383,
            ],
            [
                0.0469057130870883,
                -0.35957259110238166,
                0.06309916143856897,
                -0.3126531125455273,
            ],
            [
                -0.04527237752903268,
                -0.4646223970748078,
                0.1432359699471787,
                -0.40150947016647276,
            ],
        ]

        bboxes = to_test.bbox(self.polygons).tolist()
        for ans, bbox in zip(answer, bboxes):
            np.testing.assert_allclose(ans, bbox, rtol=RTOL, atol=ATOL)

    def test_bounding_box(self):
        with self.assertRaises(AttributeError):
            to_test.bounding_box(self.points)

        line_rects = to_test.bounding_box(self.lines).tolist()
        line_bboxes = [[(a.left, a.lower), (a.right, a.upper)] for a in line_rects]
        pgon_rects = to_test.bounding_box(self.polygons).tolist()
        pgon_bboxes = [[(a.left, a.lower), (a.right, a.upper)] for a in pgon_rects]

        line_answers = [
            [
                (-0.009053924887015952, -0.2589587703323735),
                (0.007481157395930582, -0.25832280562918325),
            ],
            [
                (0.10923550990637088, -0.2564149115196125),
                (0.12895041570526866, -0.2564149115196125),
            ],
            [
                (0.050726757212867735, -0.356261369920482),
                (0.06153815716710198, -0.3130157701035449),
            ],
            [
                (-0.0414881247497188, -0.46055958124368335),
                (0.1391258509563127, -0.4058666167693217),
            ],
        ]
        pgon_answers = [
            [
                (-0.010809397704086565, -0.26282711761789435),
                (0.12787295484449185, -0.250785835510383),
            ],
            [
                (0.0469057130870883, -0.35957259110238166),
                (0.06309916143856897, -0.3126531125455273),
            ],
            [
                (-0.04527237752903268, -0.4646223970748078),
                (0.1432359699471787, -0.40150947016647276),
            ],
        ]

        for bbox, answer in zip(line_bboxes, line_answers):
            np.testing.assert_allclose(bbox, answer, atol=ATOL, rtol=RTOL)
        for bbox, answer in zip(pgon_bboxes, pgon_answers):
            np.testing.assert_allclose(bbox, answer, atol=ATOL, rtol=RTOL)
        for rectangle in line_rects + pgon_rects:
            self.assertIsInstance(rectangle, Rectangle)

    def test_centroid(self):
        with self.assertRaises(AttributeError):
            to_test.centroid(self.points)
        with self.assertRaises(AttributeError):
            to_test.centroid(self.lines)

        centroids = to_test.centroid(self.polygons).tolist()

        centroid_answers = [
            (0.06466214975239247, -0.257330080795802),
            (0.05151163524856857, -0.33495102150875505),
            (0.04759584610455384, -0.44147205133285744),
        ]

        for ct, answer in zip(centroids, centroid_answers):
            np.testing.assert_allclose(ct, answer, rtol=RTOL, atol=ATOL)

    def test_holes(self):
        holed_polygons = rf(get_path("Polygon_Holes.shp"))
        with self.assertRaises(AttributeError):
            to_test.centroid(self.points)
        with self.assertRaises(AttributeError):
            to_test.centroid(self.lines)

        no_holes = to_test.holes(self.polygons).tolist()
        holes = to_test.holes(holed_polygons).tolist()

        for elist in no_holes:
            self.assertEqual(elist, [[]])

        answers = [
            [
                [
                    (-0.002557818613137461, -0.25599115990199145),
                    (0.0012028146993492903, -0.25561239107915107),
                    (0.004909338180001697, -0.2596435735508095),
                    (-0.0019896653788768724, -0.2616726922445973),
                    (-0.007021879739470651, -0.25834493758678534),
                    (-0.002557818613137461, -0.25599115990199145),
                ],
                [
                    (0.11456291239229519, -0.2534750527216944),
                    (0.11878347927537383, -0.2540973157877893),
                    (0.11878347927537383, -0.2540973157877893),
                    (0.12335576006537571, -0.25596410498607414),
                    (0.11605093276773958, -0.258155553175365),
                    (0.11020707092963067, -0.2579391138480276),
                    (0.11456291239229519, -0.2534750527216944),
                ],
            ],
            [
                [
                    (0.04818367618951632, -0.31403748200228154),
                    (0.052755956979518195, -0.31384809759086135),
                    (0.04975286131271223, -0.3566219196559085),
                    (0.04818367618951632, -0.31403748200228154),
                ]
            ],
            [
                [
                    (-0.039609525961703126, -0.4112999047245106),
                    (-0.013745026344887779, -0.43770550265966934),
                    (-0.015260101636249357, -0.4393287976146996),
                    (-0.04242323721708889, -0.4140053963162277),
                    (-0.039609525961703126, -0.4112999047245106),
                ],
                [
                    (0.027838379419803827, -0.4597823140480808),
                    (0.07350707748798824, -0.45859189774772524),
                    (0.07469749378834376, -0.46064807135743024),
                    (0.028487697401815927, -0.46270424496713525),
                    (0.027838379419803827, -0.4597823140480808),
                ],
                [
                    (0.11192505809037084, -0.43467535207694624),
                    (0.13962929198955382, -0.4037245282677028),
                    (0.14092792795357803, -0.405023164231727),
                    (0.11463054968208794, -0.4370561846776573),
                    (0.11192505809037084, -0.43467535207694624),
                ],
            ],
        ]
        for hole, answer in zip(holes, answers):
            for sub_hole, sub_answer in zip(hole, answer):
                np.testing.assert_allclose(sub_hole, sub_answer, rtol=RTOL, atol=ATOL)

    def test_len(self):
        with self.assertRaises(AttributeError):
            to_test.len(self.points)

        line_len = to_test.len(self.lines)
        pgon_len = to_test.len(self.polygons)

        pgon_answers = [24, 7, 14]
        line_answers = [
            0.016547307853772356,
            0.019714905798897786,
            0.058991346117778738,
            0.21634275419393173,
        ]
        np.testing.assert_allclose(line_len, line_answers, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(pgon_len, pgon_answers, rtol=RTOL, atol=ATOL)

    def test_parts(self):
        with self.assertRaises(AttributeError):
            to_test.parts(self.points)

        line_parts = to_test.parts(self.lines)
        pgon_parts = to_test.parts(self.polygons)

        pgon_answers = [
            [
                [
                    (-0.010809397704086565, -0.25825973474952796),
                    (-0.007487664708911018, -0.25493800175435244),
                    (-0.0016746319673538457, -0.2532771352567647),
                    (0.003307967525409461, -0.2545227851299555),
                    (0.006214483896188033, -0.25701408487633715),
                    (0.007044917144981927, -0.26033581787151266),
                    (0.003307967525409461, -0.26241190099349737),
                    (-0.0029202818405446584, -0.26282711761789435),
                    (-0.008318097957704912, -0.26199668436910045),
                    (-0.009978964455292672, -0.26075103449590964),
                    (-0.010809397704086565, -0.25825973474952796),
                ],
                [
                    (0.10711212362464478, -0.25618365162754325),
                    (0.1112642898686142, -0.25203148538357384),
                    (0.11583167273698053, -0.250785835510383),
                    (0.12164470547853773, -0.25203148538357384),
                    (0.12538165509811022, -0.25410756850555855),
                    (0.12787295484449185, -0.2574293015007341),
                    (0.12579687172250714, -0.26033581787151266),
                    (0.1191534057321561, -0.26116625112030656),
                    (0.1141708062393928, -0.26158146774470353),
                    (0.11084907324421728, -0.25992060124711575),
                    (0.10794255687343868, -0.25909016799832185),
                    (0.10794255687343868, -0.25909016799832185),
                    (0.10711212362464478, -0.25618365162754325),
                ],
            ],
            [
                [
                    (0.05396439570183631, -0.3126531125455273),
                    (0.05147309595545463, -0.35251390848763364),
                    (0.059777428443393454, -0.34254870950210703),
                    (0.06309916143856897, -0.34462479262409174),
                    (0.048981796209073, -0.35957259110238166),
                    (0.0469057130870883, -0.3126531125455273),
                    (0.05396439570183631, -0.3126531125455273),
                ]
            ],
            [
                [
                    (-0.04527237752903268, -0.413550752273984),
                    (-0.039874561411872456, -0.4077377195324269),
                    (-0.039874561411872456, -0.4077377195324269),
                    (-0.010809397704086565, -0.43680288324021277),
                    (0.02656009849163815, -0.45756371446005983),
                    (0.07181871055090477, -0.45673328121126594),
                    (0.1104338566198203, -0.4338963668694341),
                    (0.1394990203276062, -0.40150947016647276),
                    (0.1432359699471787, -0.4052464197860452),
                    (0.1162468893613775, -0.4380485331134035),
                    (0.07763174329246192, -0.4625463139528231),
                    (0.02780574836482899, -0.4646223970748078),
                    (-0.013715914074865138, -0.4442767824793577),
                    (-0.04527237752903268, -0.413550752273984),
                ]
            ],
        ]
        line_answers = [
            [
                [
                    (-0.009053924887015952, -0.25832280562918325),
                    (0.007481157395930582, -0.2589587703323735),
                    (0.007481157395930582, -0.2589587703323735),
                ]
            ],
            [
                [
                    (0.10923550990637088, -0.2564149115196125),
                    (0.12895041570526866, -0.2564149115196125),
                ]
            ],
            [
                [
                    (0.050726757212867735, -0.3130157701035449),
                    (0.050726757212867735, -0.356261369920482),
                    (0.06153815716710198, -0.3448140052630575),
                    (0.06153815716710198, -0.3448140052630575),
                ]
            ],
            [
                [
                    (-0.0414881247497188, -0.41286222850441445),
                    (-0.012233748402967204, -0.4402087107415953),
                    (0.027196063194828424, -0.46055958124368335),
                    (0.07489341593409732, -0.4586516871341126),
                    (0.11241533342232213, -0.43639292252245376),
                    (0.1391258509563127, -0.4058666167693217),
                ]
            ],
        ]

        for part, answer in zip(pgon_parts, pgon_answers):
            for piece, sub_answer in zip(part, answer):
                np.testing.assert_allclose(piece, sub_answer, rtol=RTOL, atol=ATOL)

    def test_perimeter(self):
        with self.assertRaises(AttributeError):
            to_test.perimeter(self.points)
        with self.assertRaises(AttributeError):
            to_test.perimeter(self.lines)

        pgon_perim = to_test.perimeter(self.polygons)
        pgon_answers = np.array([0.09381641, 0.13141213, 0.45907697])

        np.testing.assert_allclose(
            pgon_perim.values, pgon_answers, rtol=RTOL, atol=ATOL
        )

    def test_segments(self):
        with self.assertRaises(AttributeError):
            to_test.segments(self.points)
        with self.assertRaises(AttributeError):
            to_test.segments(self.polygons)

        line_segments = to_test.segments(self.lines)
        flattened = [l[0] for l in line_segments]

        answers = [
            [
                (
                    (-0.009053924887015952, -0.25832280562918325),
                    (0.007481157395930582, -0.2589587703323735),
                ),
                (
                    (0.007481157395930582, -0.2589587703323735),
                    (0.007481157395930582, -0.2589587703323735),
                ),
            ],
            [
                (
                    (0.10923550990637088, -0.2564149115196125),
                    (0.12895041570526866, -0.2564149115196125),
                )
            ],
            [
                (
                    (0.050726757212867735, -0.3130157701035449),
                    (0.050726757212867735, -0.356261369920482),
                ),
                (
                    (0.050726757212867735, -0.356261369920482),
                    (0.06153815716710198, -0.3448140052630575),
                ),
                (
                    (0.06153815716710198, -0.3448140052630575),
                    (0.06153815716710198, -0.3448140052630575),
                ),
            ],
            [
                (
                    (-0.0414881247497188, -0.41286222850441445),
                    (-0.012233748402967204, -0.4402087107415953),
                ),
                (
                    (-0.012233748402967204, -0.4402087107415953),
                    (0.027196063194828424, -0.46055958124368335),
                ),
                (
                    (0.027196063194828424, -0.46055958124368335),
                    (0.07489341593409732, -0.4586516871341126),
                ),
                (
                    (0.07489341593409732, -0.4586516871341126),
                    (0.11241533342232213, -0.43639292252245376),
                ),
                (
                    (0.11241533342232213, -0.43639292252245376),
                    (0.1391258509563127, -0.4058666167693217),
                ),
            ],
        ]

        for parts, points in zip(flattened, answers):
            for piece, answer in zip(parts, points):
                self.assertIsInstance(piece, LineSegment)
                p1, p2 = piece.p1, piece.p2
                np.testing.assert_allclose([p1, p2], answer)

    def test_vertices(self):
        with self.assertRaises(AttributeError):
            to_test.vertices(self.points)

        line_verts = to_test.vertices(self.lines).tolist()
        pgon_verts = to_test.vertices(self.polygons).tolist()

        line_answers = [
            [
                (-0.009053924887015952, -0.25832280562918325),
                (0.007481157395930582, -0.2589587703323735),
                (0.007481157395930582, -0.2589587703323735),
            ],
            [
                (0.10923550990637088, -0.2564149115196125),
                (0.12895041570526866, -0.2564149115196125),
            ],
            [
                (0.050726757212867735, -0.3130157701035449),
                (0.050726757212867735, -0.356261369920482),
                (0.06153815716710198, -0.3448140052630575),
                (0.06153815716710198, -0.3448140052630575),
            ],
            [
                (-0.0414881247497188, -0.41286222850441445),
                (-0.012233748402967204, -0.4402087107415953),
                (0.027196063194828424, -0.46055958124368335),
                (0.07489341593409732, -0.4586516871341126),
                (0.11241533342232213, -0.43639292252245376),
                (0.1391258509563127, -0.4058666167693217),
            ],
        ]
        pgon_answers = [
            [
                (-0.010809397704086565, -0.25825973474952796),
                (-0.007487664708911018, -0.25493800175435244),
                (-0.0016746319673538457, -0.2532771352567647),
                (0.003307967525409461, -0.2545227851299555),
                (0.006214483896188033, -0.25701408487633715),
                (0.007044917144981927, -0.26033581787151266),
                (0.003307967525409461, -0.26241190099349737),
                (-0.0029202818405446584, -0.26282711761789435),
                (-0.008318097957704912, -0.26199668436910045),
                (-0.009978964455292672, -0.26075103449590964),
                (-0.010809397704086565, -0.25825973474952796),
                (0.10711212362464478, -0.25618365162754325),
                (0.1112642898686142, -0.25203148538357384),
                (0.11583167273698053, -0.250785835510383),
                (0.12164470547853773, -0.25203148538357384),
                (0.12538165509811022, -0.25410756850555855),
                (0.12787295484449185, -0.2574293015007341),
                (0.12579687172250714, -0.26033581787151266),
                (0.1191534057321561, -0.26116625112030656),
                (0.1141708062393928, -0.26158146774470353),
                (0.11084907324421728, -0.25992060124711575),
                (0.10794255687343868, -0.25909016799832185),
                (0.10794255687343868, -0.25909016799832185),
                (0.10711212362464478, -0.25618365162754325),
            ],
            [
                (0.05396439570183631, -0.3126531125455273),
                (0.05147309595545463, -0.35251390848763364),
                (0.059777428443393454, -0.34254870950210703),
                (0.06309916143856897, -0.34462479262409174),
                (0.048981796209073, -0.35957259110238166),
                (0.0469057130870883, -0.3126531125455273),
                (0.05396439570183631, -0.3126531125455273),
            ],
            [
                (-0.04527237752903268, -0.413550752273984),
                (-0.039874561411872456, -0.4077377195324269),
                (-0.039874561411872456, -0.4077377195324269),
                (-0.010809397704086565, -0.43680288324021277),
                (0.02656009849163815, -0.45756371446005983),
                (0.07181871055090477, -0.45673328121126594),
                (0.1104338566198203, -0.4338963668694341),
                (0.1394990203276062, -0.40150947016647276),
                (0.1432359699471787, -0.4052464197860452),
                (0.1162468893613775, -0.4380485331134035),
                (0.07763174329246192, -0.4625463139528231),
                (0.02780574836482899, -0.4646223970748078),
                (-0.013715914074865138, -0.4442767824793577),
                (-0.04527237752903268, -0.413550752273984),
            ],
        ]
        for part, answer in zip(line_verts, line_answers):
            np.testing.assert_allclose(part, answer, atol=ATOL, rtol=RTOL)
        for part, answer in zip(pgon_verts, pgon_answers):
            np.testing.assert_allclose(part, answer, atol=ATOL, rtol=RTOL)
