^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package compressed_depth_image_transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.15.0 (2023-01-30)
-------------------
* Update maintainer in ros1 branches (`#132 <https://github.com/ros-perception/image_transport_plugins/issues/132>`_)
* Fix number of elements in vector parameter to call cv::imencode for OpenCV 4.7 (`#130 <https://github.com/ros-perception/image_transport_plugins/issues/130>`_)
* Fix uninitialized memory usage (`#125 <https://github.com/ros-perception/image_transport_plugins/issues/125>`_)
* Make the default compressed depth png_level 1 instead of 9 to save cpu (`#85 <https://github.com/ros-perception/image_transport_plugins/issues/85>`_)
  The default 9 is so cpu intensive to be frequently unusable and results in many dropped frames, and there is still plenty of bandwidth savings from the compression at level 1.
* Fix copyright year 20012 -> 2012 (`#80 <https://github.com/ros-perception/image_transport_plugins/issues/80>`_)
* Fix regression in compressed_depth_image_transport with old bags (`#64 <https://github.com/ros-perception/image_transport_plugins/issues/64>`_)
* Contributors: Johannes Meyer, Kenji Brameld, Lucas Walter, Martin Pecka, Michael Carroll, Timm Linder, ijnek, v4hn

1.14.0 (2020-05-14)
-------------------
* Bump CMake version to avoid CMP0048 warning (`#53 <https://github.com/ros-perception/image_transport_plugins/issues/53>`_)
* Add depth compression by RVL `#42 <https://github.com/ros-perception/image_transport_plugins/issues/42>`_
* Fix binary install locations for Windows build `#34 <https://github.com/ros-perception/image_transport_plugins/issues/34>`_
* Add legacy constants when using opencv4 `#32 <https://github.com/ros-perception/image_transport_plugins/issues/32>`_
* Contributors: Alejandro Hernández Cordero, David Gossow, Hans Gaiser, Sean Yen, Shuntaro Yamazaki

1.9.5 (2016-10-03)
------------------
* disable -Werr
* Contributors: Vincent Rabaud

1.9.4 (2016-10-02)
------------------
* address gcc6 build error and tune
  With gcc6, compiling fails with 'stdlib.h: No such file or directory',
  as including '-isystem /usr/include' breaks with gcc6, cf.,
  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70129.
  This commit addresses this issue for this package in the same way
  it was addressed in various other ROS packages. A list of related
  commits and pull requests is at:
  https://github.com/ros/rosdistro/issues/12783
  Signed-off-by: Lukas Bulwahn <lukas.bulwahn@oss.bmw-carit.de>
* Fix a missing return statement and add -Wall -Werror.
* Contributors: Lukas Bulwahn, Mac Mason

1.9.3 (2016-01-17)
------------------
* Refactor the codec into its own .h and .cpp.
* remove useless tf dependencies
* Contributors: Mac Mason, Vincent Rabaud

1.9.2 (2015-04-25)
------------------
* use compression parameters for both depths
  fixes `#12 <https://github.com/ros-perception/image_transport_plugins/issues/12>`_
* get code to compile with OpenCV3
* Contributors: Vincent Rabaud

1.9.1 (2014-07-18)
------------------

1.9.0 (2014-05-16)
------------------

1.8.21 (2013-06-27)
-------------------
* maintainer: david gossow
* Contributors: David Gossow

1.8.20 (2013-03-18)
-------------------
* 1.8.19 -> 1.8.20
* Contributors: Julius Kammerl

1.8.19 (2013-02-24)
-------------------
* 1.8.18 -> 1.8.19
* Contributors: Julius Kammerl

1.8.18 (2013-02-07 17:59)
-------------------------
* 1.8.17 -> 1.8.18
* fixing input format checks (enabling rgba, bgra) + minor fixes
* Contributors: Julius Kammerl

1.8.17 (2013-01-18)
-------------------
* 1.8.16 -> 1.8.17
* Contributors: Julius Kammerl

1.8.16 (2013-01-17)
-------------------
* 1.8.15 -> 1.8.16
* use the pluginlib script to remove some runtime warnings
* Contributors: Julius Kammerl, Vincent Rabaud

1.8.15 (2012-12-28 20:11)
-------------------------

1.8.14 (2012-12-28 20:02)
-------------------------

1.8.13 (2012-12-28 19:06)
-------------------------
* fix the bad exports
* make sure the plugins are visible by image_transport
* added license headers to various cpp and h files
* Contributors: Aaron Blasdel, Vincent Rabaud

1.8.12 (2012-12-19 19:30)
-------------------------
* fix downstream stuff in cmake
* Contributors: Dirk Thomas

1.8.11 (2012-12-19 17:17)
-------------------------
* fix cmake order
* Contributors: Dirk Thomas

1.8.10 (2012-12-19 17:03)
-------------------------
* fix dyn reconf
* Contributors: Dirk Thomas

1.8.9 (2012-12-19 00:26)
------------------------
* switching to verion 1.8.9
* Contributors: Julius Kammerl

1.8.8 (2012-12-17)
------------------
* adding build_deb on message_generation & mrun_deb on message_runtime
* Updated package.xml for new buildtool_depend tag for catkin requirement
* Contributors: Julius Kammerl, mirzashah

1.8.7 (2012-12-10 15:29)
------------------------
* adding missing tf build dependency
* Contributors: Julius Kammerl

1.8.6 (2012-12-10 15:08)
------------------------
* switching to version 1.8.6
* Contributors: Julius Kammerl

1.8.5 (2012-12-09)
------------------
* adding missing build debs
* added class_loader_hide_library_symbols macros to CMakeList
* switching to 1.8.5
* Contributors: Julius Kammerl

1.8.4 (2012-11-30)
------------------
* switching to version 1.8.4
* adding plugin.xml exports for pluginlib
* catkinizing theora_image_transport
* catkinizing compressed_depth_image_transport
* github migration from code.ros.org (r40053)
* Contributors: Julius Kammerl
