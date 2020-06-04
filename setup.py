from setuptools import setup

# import ``__version__`` from code base
exec(open('MPLearn/version.py').read())

setup(
    name='MPLearn',
    version=__version__,
    description='Machine learning methods for analyzing Morphological Profiling data',
    author="Matthew O'Meara",
    author_email="maom@umich.edu",
    packages=['MPLearn',],
    install_requires=[
        'numpy>=1.12.1',
        'six>=1.10.0'],
    extras_require={
        'boto3' : ['boto3>=1.13.15'],
        'bokeh' : ['bokeh>=1.0.0'],
        'dask' : ['dask>=0.19.1'],
        'joblib' : ['0.12.5'],        
        'datashader' : ['datashader>=0.5.4'],
        'distributed' : ['distributed>=1.23.1'],
        'holovoiews' : ['holoviews>=1.10.9'],
        'pandas' : ['pandas>=1.0.1'],
        'mysql' : ['mysql>=5.7.20'],
        'umap-learn' : ['umap-learn>0.3.6'],
    },
    scripts=['bin/embed_umap'],
    tests_require=['pytest'],
    url='http://github.com/momeara/MPLearn',
    keywords='machine learning CellPaining',
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
