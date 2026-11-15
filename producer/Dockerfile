# Producer image: Ubuntu 24.04 + ROS 2 Jazzy. Reads the dataset and publishes it.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY producer src/producer
RUN . /opt/ros/jazzy/setup.sh && colcon build && rm -rf build log
COPY data /data

CMD ["bash", "-c", ". /ws/install/setup.bash && ros2 run producer producer_node --ros-args --params-file /ws/install/producer/share/producer/config/producer.yaml"]
