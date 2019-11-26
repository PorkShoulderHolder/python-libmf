import libmf
print(libmf.__file__)
from libmf import mf
from unittest import TestCase, main
import math


class TestMF(TestCase):
    mf_engine = mf.MF()
    test_data = mf.generate_test_data(100, 50, 1000)
    mf_engine.fit(test_data)

    def test_fit(self):
        print("testing fit method")
        TestMF.mf_engine.fit(TestMF.test_data)

    def test_predict(self):
        print("testing predict method")
        n = 1000
        pred_data = mf.generate_test_data(100, 50, n, indices_only=True)
        predictions = TestMF.mf_engine.predict(pred_data)
        self.assertTrue(len(predictions) == n)
        for i in predictions.tolist():
            self.assertTrue(type(i) is float)
            self.assertFalse(math.isnan(i))

    def test_cross_val(self):
        print("testing cross val method")
        val_data = mf.generate_test_data(100, 50, 1000)
        TestMF.mf_engine.mf_cross_validation(val_data)

    def test_factorq(self):
        print("testing factor-q")
        q = TestMF.mf_engine.q_factors()
        self.assertEqual(q.shape, (TestMF.mf_engine.model.n, TestMF.mf_engine.model.k))

    def test_factorp(self):
        print("testing factor-p")
        p = TestMF.mf_engine.p_factors()
        self.assertEqual(p.shape, (TestMF.mf_engine.model.m, TestMF.mf_engine.model.k))


if __name__ == '__main__':
    main()

