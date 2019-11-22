from distutils.core import setup
from setuptools import find_packages


setup(name='dymos',
    version='0.14.2',
    description='Open-Source Optimization of Dynamic Multidisciplinary Systems',
    url='https://github.com/OpenMDAO/dymos',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    license='Apache License',
    packages=find_packages(),
    install_requires=[
        'openmdao>=2.9.0',
        'numpy>=1.14.1',
        'scipy>=1.0.0',
        'six',
        'pep8',
        'parameterized',
        'sphinx',
        'sphinxcontrib-bibtex',
        'redbaron'
    ],
    zip_safe=False,
)
