# Consumer image: Ubuntu 24.04 + ROS 2 Jazzy + scientific stack. Runs the processing pipeline.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-scipy python3-matplotlib python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY consumer src/consumer
RUN . /opt/ros/jazzy/setup.sh && colcon build && rm -rf build log
COPY tests tests

ENV MPLBACKEND=Agg
# ros2 launch exits 0 even if a node failed, so success = a fresh results.json (the report deletes stale ones).
CMD ["bash", "-c", ". /ws/install/setup.bash && ros2 launch consumer pipeline.launch.py && test -f /output/results.json"]
