from setuptools import setup

setup(
    name='machine-learning',
    version='1.0.0',
    packages=['machine-learning', 'service', 'preprocessor', 'trainer', 'utils', 'bin', 'stop_words'],
    package_dir={
        'machine-learning': 'app',
        'bin': 'resources\\bin',
        'stop_words': 'resources\\stop_words',
        'service': 'app\\service',
        'preprocessor': 'app\\preprocessor',
        'trainer': 'app\\trainer',
        'utils': 'app\\utils'
    },
    package_data={
        'bin': ['*.bin'],
        'stop_words': ['*.txt'],
    },
    url='https://omniway.ua',
    license='Omniway',
    author='Evgeniy Chumak',
    author_email='evgeniy.chumak@omniway.ua',
    description='Machine learning tools',
    python_requires='==3.8',
    platform='win-amd64'
)
