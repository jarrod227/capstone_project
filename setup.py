from setuptools import setup, find_packages

setup(
    name="eog-cursor-control",
    version="1.0.0",
    description="EOG-based cursor control system using eye movement and head motion",
    author="Jarrod",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.10",
    install_requires=[
        "pyserial>=3.5",
        "pyautogui>=0.9.54",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "pandas>=2.0.0",
        "pynput>=1.7.6",
    ],
)
