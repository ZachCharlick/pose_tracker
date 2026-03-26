from setuptools import find_packages, setup

package_name = 'pose_landmarker_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zacharycharlick',
    maintainer_email='zacharycharlick@gmail.com',
    description='MediaPipe Pose Landmarker ROS 2 node.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_landmarker_node = pose_landmarker_ros.pose_landmarker_node:main',
            'forearm_pose_3d_node = pose_landmarker_ros.forearm_pose_3d_node:main',
        ],
    },
)
