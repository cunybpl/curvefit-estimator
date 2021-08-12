#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get VERSION from _version.py
ver_file = os.path.join('curvefit', '_version.py')
with open(ver_file) as f:
    version={}
    exec(f.read(), version)

DISTNAME = 'curvefit'
DESCRIPTION = 'A sckit-learn wrapper around scipy.optimize.curve_fit'
with codecs.open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'cunybpl'

MAINTAINER_EMAIL = ''
URL = 'https://github.com/cunybpl/curvefit-estimator.git'

LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/cunybpl/curvefit-estimator.git'

INSTALL_REQUIRES = [
    'scikit-learn>=0.24'
]

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7', 
               'Programming Language :: Python :: 3.8']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=version['VERSION'],
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(exclude=["tests"]),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE
    )
