from distutils.core import setup, Extension
import unittest


def gen_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='*_test.py')
    return test_suite


setup(name='libmf',
      packages=['libmf'],
      version='0.9.2',
      test_suite='setup.gen_test_suite',
      description='python bindings to libmf',
      author='Sam Fox Royston (bindings), Chih-Jen Lin (c++ source)',
      author_email='sfoxroyston@gmail.com',
      license="MIT",
      url='https://github.com/PorkShoulderHolder/python-libmf',
      headers=['src_cpp/mf.h'],
      install_requires=['numpy'],
      ext_modules=[Extension('libmf',
                             sources=['src_cpp/libmf_interface.cpp', 'src_cpp/mf.cpp'],
                             depends=['src_cpp/mf.h'],
                             extra_compile_args=['-std=c++11'])
                   ],
      keywords=["matrix factorization", "machine learning", "parallelism", "python", "out of core"],
      )
