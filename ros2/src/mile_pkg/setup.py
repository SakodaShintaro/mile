from setuptools import setup

package_name = 'mile_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sakoda',
    maintainer_email='sakoda@keio.jp',
    description='MILE Package',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "mile_node = mile_pkg.main:main"
        ],
    },
)
