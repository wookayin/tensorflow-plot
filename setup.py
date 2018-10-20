#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import re
import sys
from setuptools import setup, Command


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def read_version():
    # importing the package causes an ImportError :-)
    with open(os.path.join(__PATH__, 'tfplot/__init__.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")

__version__ = read_version()
readme = codecs.open('README.md', encoding='utf-8').read()


# brought from https://github.com/kennethreitz/setup.py
class DeployCommand(Command):
    description = 'Build and deploy the package to PyPI.'
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    @staticmethod
    def status(s):
        print(s)

    def run(self):
        import twine  # we require twine locally

        assert 'dev' not in __version__, \
            "Only non-devel versions are allowed. __version__ == {}".format(__version__)

        with os.popen("git status --short") as fp:
            git_status = fp.read().strip()
            if git_status:
                print("Error: git repository is not clean.\n")
                os.system("git status --short")
                sys.exit(1)

        try:
            from shutil import rmtree
            self.status('Removing previous builds ...')
            rmtree(os.path.join(__PATH__, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution ...')
        os.system('{0} setup.py sdist'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine ...')
        ret = os.system('twine upload dist/*')
        if ret != 0:
            sys.exit(ret)

        self.status('Creating git tags ...')
        os.system('git tag v{0}'.format(read_version()))
        os.system('git tag --list')
        sys.exit()


setup(
    name='tensorflow-plot',
    version=__version__,
    description='TensorFlow Plot',
    long_description=readme,
    license='MIT License',
    url='https://github.com/wookayin/tensorflow-plot',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    keywords='tensorflow matplotlib tensorboard plot tfplot',
    packages=[
        'tfplot',
    ],
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'six',
        'numpy',
        'matplotlib>=2.0.0',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
    cmdclass={
        'deploy': DeployCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
