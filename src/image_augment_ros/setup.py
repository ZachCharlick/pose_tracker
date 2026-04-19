from setuptools import find_packages, setup

package_name = 'image_augment_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy<2.0.0'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Camera image augmentation node for ROS 2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'image_augment_node = image_augment_ros.image_augment_node:main',
        ],
    },
)
