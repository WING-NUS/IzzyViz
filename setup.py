from setuptools import setup, find_packages

setup(
    name='IzzyViz',
    version='0.1.0',
    description='A library for visualizing attention scores in transformer models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Luo Xizi',
    author_email='e0909010@u.nus.edu',
    url='https://github.com/lxz333/IzzyViz',  # Update with your GitHub URL
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.0.0',
        'numpy>=1.15.0',
        'torch>=1.0.0',
        'seaborn>=0.9.0',
        'transformers>=4.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose an appropriate license
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
    license='MIT',  # Match this with your LICENSE file
)
