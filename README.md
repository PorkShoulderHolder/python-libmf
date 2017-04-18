## compile libmf interface object files

you just need a standard c++ compiler 
```
$ g++ --std=c++11 *.cpp -shared -o libmf.so
```

that should create an libmg.so file which mf.py will use the interface of
