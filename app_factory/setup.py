from setuptools import setup

setup(
    name='app_factory',
    version='0.1',
    py_modules=['app_factory'],
    install_requires=[
        'Click',
        'httpx',
    ],
    entry_points='''
        [console_scripts]
        app_factory=app_factory:sync_app_factory_wrapper
    ''',
)
