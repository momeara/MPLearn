from setuptools import setup

# import ``__version__`` from code base
exec(open('MPLearn/version.py').read())

setup(
    name='MPLearn',
    version=__version__,
    description='Machine learning methods for analyzing Morphological Profiling data',
    author="Matthew O'Meara",
    author_email="maom@umich.edu",
    packages=['MPLearn', 'MPLearn.chemoinformatics'],
    install_requires=[
        'numpy>=1.12.1',
        'six>=1.10.0',
        'joblib>=1.0.1',
        'pyarrow>=3.0.0',
        'pandas>=1.2.3',
        'scikit-learn>=0.24.1',
        'umap-learn>=0.3.6'],
    extras_require={
        'boto3' : ['boto3>=1.13.15'],
        'bokeh' : ['bokeh>=1.0.0'],
        'dask' : ['dask>=0.19.1'],
        'datashader' : ['datashader>=0.5.4'],
        'distributed' : ['distributed>=1.23.1'],
        'holovoiews' : ['holoviews>=1.10.9'],
        'mysql' : ['mysql>=5.7.20'],
    },
    scripts=['bin/embed_umap', 'bin/featurize_substances'],
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
