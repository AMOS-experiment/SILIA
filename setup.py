import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SILIA", # Replace with your own username
    version="0.1",
    author="Amrut Nadgir",
    author_email="amrut.nadgir@gmail.com",
    description="A software implementation of a multi-channel and multi-frequency lock-in amplifier to extract periodic features from data.",
    url="https://github.com/amrutn/SILIA",
    packages=['SILIA',],
    license='LICENSE.txt',
    long_description=long_description,
    install_requires=[
       "numpy==1.15.4",
       "scipy==1.2.1",
       "tqdm",
   ],
   python_requires='>=3'
)