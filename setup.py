from setuptools import setup, find_packages

setup(
    name='IzzyViz',
    version='0.1.0',
    description='A library for visualizing attention scores in transformer models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Luo Xizi',
    author_email='e0909010@u.nus.edu',
    url='https://github.com/lxz333/IzzyViz',  
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.0.0',
        'numpy>=1.15.0,<2.0.0',  # Limit NumPy version to avoid issues with NumPy 2.1+
        'torch>=1.0.0',
        'transformers>=4.0.0',
        'pandas>=1.4.0',
        'pybind11>=2.12'  # Add pybind11 if compilation might be needed for compatibility
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
    license='MIT', 
)
