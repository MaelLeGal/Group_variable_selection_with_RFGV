import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open('requirements.txt', "r", encoding="utf-8") as req_file:       
	requirements = [req.strip() for req in req_file.read().splitlines()
	
setuptools.setup(
    name="rfgi",
    version="0.0.7",
    author="Maël Le Gal, Audrey Poterie, Charlotte Pelletier",
    author_email="mael.legal@live.fr",
    description="Random Forest for Grouped Inputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apoterie/Group_variable_selection_with_RFGV",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
	install_requires= [
		"numpy",
		"pandas",
		"matplotlib",
		"scikit-learn",
		"jupyter",
		"scipy",
		"joblib"
	]
)