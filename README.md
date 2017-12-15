## install with pip
```
pip install libmf
```

#### Or install with setup.py
```
python setup.py install
```

#### Or compile from source

Still easy, you just need a standard c++ compiler, and you need to make sure that your mf.py file can find the libmf.so
 file
```
$ cd python-libmf
$ g++ --std=c++11 src/*.cpp -shared -o libmf.so
```

That should create a python-libmf.so file which mf.py will use to interface with libmf.
Make sure you know where this file is, because mf.py needs to reference it.

After compilation try running:
```
$ python tests/mf_tests.py
```

if these work then you are good to go!

```
>>> from libmf import mf
>>> engine = mf.MF()
>>> engine.fit(data)
>>> engine.dict(ind)
```
`data` is a sparse numpy array consisting of data matrix indices x and y and a corresponding value. So each row is: (x,y,v).
`data.shape => (x, 3)` where x is the number of observations

`ind` is a sparse numpy array of indices specifying where we want to predict unobserved values
