from distutils.core import setup, Extension

setup(name='python-libmf',
      version='0.1',
      description='python bindings to libmf',
      author='Sam Fox Royston',
      author_email='sfoxroyston@gmail.com',
      url='',
      packages=['libmf'],
      ext_modules=[Extension('python-libmf',
                             ['libmf_interface.cpp', 'mf.cpp'],
                             extra_compile_args=['-std=c++11'])],
     )
