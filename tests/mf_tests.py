import libmf
print libmf.__file__
from libmf import mf
from unittest import TestCase, main
import math


class TestMF(TestCase):
    mf_engine = mf.MF()
    test_data = mf.generate_test_data(9000, 1000, 25000)
    mf_engine.mf_fit(test_data)

    def test_fit(self):
        print "testing fit method"
        TestMF.mf_engine.mf_fit(TestMF.test_data)

    def test_predict(self):
        print "testing predict method"
        n = 1000
        pred_data = mf.generate_test_data(9000, 1000, n, indices_only=True)
        predictions = TestMF.mf_engine.mf_predict(pred_data)
        self.assertTrue(len(predictions) == n)
        for i in predictions.tolist():
            self.assertTrue(type(i) is float)
            self.assertFalse(math.isnan(i))

    def test_cross_val(self):
        print "testing cross val method"
        val_data = mf.generate_test_data(9000, 1000, 10000)
        TestMF.mf_engine.mf_cross_validation(val_data)


if __name__ == '__main__':
    main()

