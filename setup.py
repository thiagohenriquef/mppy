from distutils.core import setup
setup(
    name = 'mppy',
    packages = ['mppy'], # this must be the same as the name above
    version = '0.3',
    description = 'Multidimensional Projection in Python',
    author = 'Thiago Henrique Ferreira',
    author_email = 'thiagohferreira10@gmail.com',
    url = 'https://github.com/thiagohenriquef/mppy',
    download_url = 'https://github.com/thiagohenriquef/mppy/archive/0.3.tar.gz', # I'll explain this in a second
    keywords = ['multidimensional projection', 'data visualization', 'dimensionality reduction'], # arbitrary keywords
    classifiers=['Development Status :: 4 - Beta','Programming Language :: Python :: 3'],
    install_requires = ['numpy>=1.11.0', 'scipy>=0.17.0', 'matplotlib>=1.5.1', 'scikit-klearn>=0.18.1'],
)