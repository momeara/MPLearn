from setuptools import setup

# import ``__version__`` from code base
exec(open('MPLearn/version.py').read())

setup(
    name='MPLearn',
    version=__version__,
    description='Machine learning methods for analyzing Morphological Profiling data',
    author="Matthew O'Meara",
    author_email="maom@umich.edu",
    packages=['MPlearn',],
    install_requires=[
        'numpy>=1.12.1',
        'six>=1.10.0'],
    extras_require={
        'bokeh' : ['bokeh>=1.0.0'],
        'dask' : ['dask>=0.19.1'],
        'datashader' : ['datashader>=0.5.4'],
        'distributed' : ['distributed>=1.23.1'],
        'holovoiews' : ['holoviews>=1.10.9'],
        'rdkit' : ['rdkit>=2018.09.1.0'],
        'umap-learn' : ['umap-learn>0.3.6'],
    },
    tests_require=['pytest'],
    url='http://github.com/momeara/mimic',
    keywords='machine learning tensorflow molecular simulations prediction',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6'],
)
