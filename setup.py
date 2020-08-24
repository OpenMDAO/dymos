from distutils.core import setup
from setuptools import find_packages

# Setup optional dependencies
optional_dependencies = {
    'docs': [
        'mkdocs',
        'mkdocs-material',
        'pymdown-extensions',
        'mkdocs-macros-plugin',
        'mkdocs-bibtex'
    ],
    'test': [
        'pep8',
        'numpydoc>=0.9.1',
        'parameterized',
        'testflo>=1.3.6',
        'matplotlib'
    ]
}

# Add an optional dependency that concatenates all others
optional_dependencies['all'] = sorted([
    dependency
    for dependencies in optional_dependencies.values()
    for dependency in dependencies
])


setup(name='dymos',
    version='0.16.0-dev',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    license='Apache License',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'openmdao>=3.2.1',
        'numpy>=1.14.1',
        'scipy>=1.0.0'
    ],
    extras_require=optional_dependencies,
    zip_safe=False,
    package_data={'dymos.examples.aircraft_steady_flight.aero': ['data/CRM_aero_inputs.dat', 'data/CRM_aero_outputs.dat']},
    entry_points={
      'console_scripts': ['dymos=dymos.utils.command_line:dymos_cmd'],
    }
)
