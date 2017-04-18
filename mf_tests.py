from libmf import mf


test_data = mf.__generate_test_data(9000, 1000, 25000)

print "testing fit method"
mf_engine = mf.MF()
try:
	mf_engine.mf_fit(test_data)
	print "OK"
except Exception as e:
	print e
	print "FAILED"

print "testing predict method"
try:
	pred_data = mf.__generate_test_data(9000, 1000, 1000, indices_only=True)
	print pred_data.shape
	mf_engine.mf_predict(pred_data)
	print "OK"
except Exception as e:
	print "FAILED"
	print e

print "testing cross validation"
try:
	mf_engine.mf_cross_validation(test_data)
	print "OK"
except Exception as e:
	print "FAILED"
	print e
