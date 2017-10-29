__version__ = '0.2.0.dev0'

import codecs
from setuptools import setup

readme = codecs.open('README.md', encoding='utf-8').read()


setup(
    name='tfplot',
    version=__version__,
    description='TensorFlow Plot',
    long_description=readme,
    url='https://github.com/wookayin/tensorflow-plot',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    keywords='tensorflow matplotlib tensorboard plot tfplot',
    packages=[
        'tfplot',
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
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    license='MIT License',
)
