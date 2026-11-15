from setuptools import setup

STAGES = ["centerline_node", "raceline_node", "speed_profile_node", "report_node"]

setup(
    name="consumer",
    version="0.1.0",
    packages=["consumer", "consumer.pipeline"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/consumer"]),
        ("share/consumer", ["package.xml"]),
        ("share/consumer/launch", ["launch/pipeline.launch.py"]),
        ("share/consumer/config", ["config/pipeline.yaml", "config/launch.yaml"]),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [f"{s} = consumer.pipeline.{s}:main" for s in STAGES]},
)
