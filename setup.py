from setuptools import setup, find_packages

setup(
    name='coeirocore',
    version='1.1.0',
    url="https://github.com/cyrne1-7208/coeiroink_core",
    author="shirowanisan",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # 凍結版ESPnetと依存パッケージの組み合わせを検証済みのPython 3.12に固定します。
    python_requires=">=3.12,<3.13",
)
