from unittest import TestCase
TestCase.maxDiff = None
import pysal_examples

class TestHelpers(TestCase):
    def test_available(self):
        examples = pysal_examples.available()
        self.assertEqual(examples, ['10740', 'Line', 'Point', 'Polygon',
                                    'Polygon_Holes', 'arcgis', 'baltim',
                                    'book', 'burkitt', 'calemp', 'chicago',
                                    'columbus', 'desmith', 'geodanet',
                                    'juvenile', 'mexico', 'nat', 'networks',
                                    'newHaven', 'nyc_bikes', 'sacramento2',
                                    'sids2', 'snow_maps', 'south', 'stl',
                                    'street_net_pts', 'taz', 'us_income',
                                    'virginia', 'wmat'])
    def test_explain(self):
        des = 'Homicides and selected socio-economic characteristics for counties surrounding St Louis, MO. Data aggregated for three time periods: 1979-84 (steady decline in homicides), 1984-88 (stable period), and 1988-93 (steady increase in homicides).'
        e = pysal_examples.explain('stl')

        self.assertEqual(e['description'], des)
        self.assertEqual(len(e['explanation']), 10)


if __name__ == '__main__':
    unittest.main()
