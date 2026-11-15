from setuptools import setup

setup(
    name="producer",
    version="0.1.0",
    packages=["producer"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/producer"]),
        ("share/producer", ["package.xml"]),
        ("share/producer/config", ["config/producer.yaml"]),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["producer_node = producer.producer_node:main"]},
)
