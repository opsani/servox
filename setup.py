# This setup.py is required to make type annotations available to dependent projects' mypy.
# Its metadata is nonauthoritative.
from distutils.core import setup

setup(
    name="servox",
    author="Blake Watters <blake@opsani.com>",
    version = "0.11.0",
    package_data={"servo": ["py.typed"]},
    packages=["servo"]
)
