import setuptools

__version__ = '0.0.1'

REPO_NAME = "Machine Translation"
AUTHOR_USER_NAME = "danh12004"
SRC_REPO = "transformer-MT"
AUTHOR_EMAIL = "ndanh5676@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for transformer app",
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)