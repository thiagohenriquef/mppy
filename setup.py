from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import sys

if sys.version_info[:2] < (3,5):
    print("requires python 3.5 and up")

class build_ext(build_ext):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


class CTypes(Extension): pass

setup(
    name = 'mppy',
    packages = ['mppy'], # this must be the same as the name above
    version = '0.4.0rc',
    description = 'Multidimensional Projection in Python',
    ext_modules=[CTypes('force', sources=['mppy/src/force.c']), CTypes('sammon', sources=['mppy/src/sammon.c']), CTypes('kruskal', sources=['mppy/src/kruskal.c']), CTypes('lsp', sources=['mppy/src/lsp.c'])],
    cmdclass={'build_ext': build_ext},
    long_description= 'The mppy is a multidimensional projection library that generates 2D representations of high dimensional data sets.',
    author = 'Thiago Henrique Ferreira',
    author_email = 'thiagohferreira10@gmail.com',
    url = 'https://github.com/thiagohenriquef/mppy',
    download_url = 'https://github.com/thiagohenriquef/mppy/archive/0.4.0rc.tar.gz',
    keywords = ['multidimensional projection', 'data visualization', 'dimensionality reduction'],
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6'],
    #install_requires = ['numpy', 'scipy', 'matplotlib', 'scikit-learn'],
)
