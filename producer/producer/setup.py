from setuptools import setup

NODES = ["track_producer_node", "lidar_producer_node"]

setup(
    name="producer",
    version="0.1.0",
    packages=["producer"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/producer"]),
        ("share/producer", ["package.xml"]),
        ("share/producer/launch", ["launch/producer.launch.py"]),
        ("share/producer/config", ["config/producer.yaml", "config/launch.yaml"]),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [f"{n} = producer.{n}:main" for n in NODES]},
)
