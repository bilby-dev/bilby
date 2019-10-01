from __future__ import absolute_import, division

import unittest

from bilby.core import utils


class TestInferParameters(unittest.TestCase):

    def setUp(self):
        def source_function1(freqs, a, b: int):
            return None

        def source_function2(freqs, a, b, *args, **kwargs):
            return None

        def source_function3(freqs, a, b: int, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b: int, *args, **kwargs):
                pass

        class TestClass2:
            def test_method(self, freqs, a, b: int, *args, **kwargs):
                pass

        self.source1 = source_function1
        self.source2 = source_function2
        self.source3 = source_function3
        test_obj = TestClass()
        self.source4 = test_obj.test_method
        test_obj2 = TestClass()
        self.source5 = test_obj2.test_method

    def tearDown(self):
        del self.source1
        del self.source2
        del self.source3
        del self.source4

    def test_type_hinting(self):
        expected = ['a', 'b']
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_args_kwargs_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_parameters_from_function(self.source2)
        self.assertListEqual(expected, actual)

    def test_both(self):
        expected = ['a', 'b']
        actual = utils.infer_parameters_from_function(self.source3)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_args_from_method(self.source4)
        self.assertListEqual(expected, actual)

    def test_self_handling_method_as_function(self):
        expected = ['a', 'b']
        actual = utils.infer_args_from_method(self.source5)
        self.assertListEqual(expected, actual)
