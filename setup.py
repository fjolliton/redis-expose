from setuptools import setup
from pathlib import Path

long_description = Path('README.rst').read_text()

setup(
    name='redis-expose',
    version='0.2',
    description='Show the content of a Redis database',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/fjolliton/redis-expose',
    author='Frédéric Jolliton',
    author_email='github@frederic.jolliton.com',
    maintainer='Frédéric Jolliton',
    maintainer_email='github@frederic.jolliton.com',
    keywords=['Redis'],
    license='GNU General Public License (GPL)',
    packages=['redis_expose'],
    entry_points={'console_scripts': ['redis-expose=redis_expose.main:main']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Database',
        'Topic :: Database :: Front-Ends',
    ],
    install_requires=['redis'],
    extras_require={'dev': ['mypy', 'pytest']}
)
