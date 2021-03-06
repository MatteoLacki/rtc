from setuptools import setup, find_packages

setup(
    name='rtc',
    packages=find_packages(),
    version='0.0.1',
    description='Retention Time Clustering for LC-IM-MS.',
    long_description='The project implements advanced algorithms for clustering of aligned retention times in the LC-IM-MS experiments.',
    author=u'Mateusz Krzysztof Łącki',
    author_email='matteo.lacki@gmail.com',
    url='https://github.com/MatteoLacki/rtc',
    # download_url='https://github.com/MatteoLacki/MassTodonPy/tree/GutenTag',
    keywords=[
        'Analitical Chemistry',
        'Mass Spectrometry',
        'Retention Time Clustering'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6'],
    install_requires=[
        'numpy',
        'sklearn',
        'pandas',
        'scipy',
        'matplotlib',
    ]
    # scripts=[
    #     'bin/rtc']
)
