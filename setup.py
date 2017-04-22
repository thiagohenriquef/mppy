from distutils.core import setup
setup(
    name = 'mppy',
    packages = ['mppy'], # this must be the same as the name above
    version = '0.3.4',
    description = 'Multidimensional Projection in Python',
    long_description= 'The mppy is a multidimensional projection library that generates 2D representations of high dimensional data sets.',
    author = 'Thiago Henrique Ferreira',
    author_email = 'thiagohferreira10@gmail.com',
    url = 'https://github.com/thiagohenriquef/mppy',
    download_url = 'https://github.com/thiagohenriquef/mppy/archive/0.3.4.tar.gz',
    keywords = ['multidimensional projection', 'data visualization', 'dimensionality reduction'],
    classifiers=['Development Status :: 4 - Beta','Programming Language :: Python :: 3.5', 'Programming Language :: Python :: 3.6'],
    install_requires = ['numpy>=1.11.0', 'scipy>=0.17.0', 'matplotlib>=1.5.1', 'scikit-learn>=0.17'],
)
