from setuptools import setup, find_packages

setup(
    name='gpu-monitor',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['psutil', 'pynvml'],
    entry_points={
        'console_scripts': [
            'gpu-monitor=gpu_monitor.cli:main',
        ],
    },
    description='A CLI tool for monitoring GPU and system metrics.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/gpu-monitor',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
