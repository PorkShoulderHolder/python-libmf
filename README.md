## compile libmf interface object files

You just need a standard c++ compiler
```
$ g++ --std=c++11 *.cpp -shared -o python-libmf.so
```

That should create a python-libmf.so file which mf.py will use to interface with libmf.
Make sure you know where this file is, because mf.py needs to reference it.

After compilation try running:
```
$ python mf_tests.py
```

if these work then you are good to go!

```
>>> from libmf import mf
>>> engine = mf.MF()
>>> engine.mf_fit(data)
>>> engine.mf_predict(ind)
```
`data` is a sparse numpy array consisting of data matrix indices x and y and a corresponding value. So each row is: (x,y,v).
`data.shape => (x, 3)` where x is the number of observations

`ind` is a sparse numpy array of indices specifying where we want to predict unobserved values

## for mac os 10.12
```
export MACOSX_DEPLOYMENT_TARGET=10.12
```
