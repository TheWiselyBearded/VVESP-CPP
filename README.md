# VVESP Source

This directory contains the source code for the dynamically linked libraries used in the VVESP repository/project. 
Build on top of Marek Šimoník's Record3D application and source code, this extends functionality to be more performant and support various platforms such as Android, Windows, and Mac. 

To build out this library, use the following sample commands.

For Android:

`cmake -S . -B build -DANDROID_ABI="arm64-v8a" -DCMAKE_ANDROID_NDK="/Applications/Unity/Hub/Editor/2022.3.13f1/PlaybackEngines/AndroidPlayer/NDK" -DCMAKE_TOOLCHAIN_FILE="/Applications/Unity/Hub/Editor/2022.3.13f1/PlaybackEngines/AndroidPlayer/NDK/build/cmake/android.toolchain.cmake" -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_shared`

`cmake --build build`

For Mac and Windows:

`mkdir build && cd build`

`cmake ..`

`make`

(Presuming you've installed Android SDK via Unity Hub Installer)
