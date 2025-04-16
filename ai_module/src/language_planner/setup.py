import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'language_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker',
    maintainer_email='nwzantout@gmail.com',
    description='The language planner package.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'language_planner_node = language_planner.language_planner_node:main',
            'language_query_publisher = language_planner.language_query_publisher:main'
        ],
    },
)
