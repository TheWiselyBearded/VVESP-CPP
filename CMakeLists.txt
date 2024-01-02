cmake_minimum_required(VERSION 3.13)
project(record3d_unity_playback)
set(CMAKE_CXX_STANDARD 14)

if(UNIX AND NOT APPLE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Add LZFSE
if(WIN32)
    set(LZFSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/win/lzfse-lzfse-1.0)
else()
    set(LZFSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/unix/lzfse-lzfse-1.0)
endif()
add_subdirectory(${LZFSE_DIR})
include_directories(${LZFSE_DIR}/src/lzfse)


# Set the paths for libomp globally
link_directories("/opt/homebrew/opt/libomp/lib")
include_directories("/opt/homebrew/opt/libomp/include")

# Find OpenMP package
# set(OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -Xpreprocessor -fopenmp")
# set(OpenMP_C_LIB_NAMES "omp")
# set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
# set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -Xpreprocessor -fopenmp")
# set(OpenMP_CXX_LIB_NAMES "omp")
# set(OpenMP_omp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(${TARGET_NAME} OpenMP::OpenMP_CXX)
# endif()

########################################################################
### Create C++ library
########################################################################

# Define source files
set(SOURCE_FILES
    src/Record3DUnityPlayback.cpp
    src/JPEGDecoder.cpp
)

# Define include directories
include_directories(include)

# Define target name
set(TARGET_NAME record3d_unity_playback)

# Create shared library target
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

if (WIN32)
    add_compile_definitions(WIN32)
endif()

if (APPLE)
    target_link_directories(${TARGET_NAME} PUBLIC /usr/local/lib)
endif()

# Link with dependencies
target_link_libraries(${TARGET_NAME} lzfse)

# Install targets and files
include(GNUInstallDirs)
install(TARGETS ${TARGET_NAME} lzfse
        EXPORT Record3DUnityPlaybackConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT Record3DUnityPlaybackConfig DESTINATION share/Record3DUnityPlayback/cmake)

# Skip installing dependencies
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

# Android-specific settings
# if(ANDROID)
#     # Set the Android ABI (e.g., arm64-v8a)
#     set(ANDROID_ABI "arm64-v8a")

#     # Set the Android NDK path
#     set(ANDROID_NDK "/Users/alirezabahremand/Library/Android/sdk/ndk")

#     # Set the Android platform version
#     set(ANDROID_PLATFORM "android-21")

#     # Set the Android STL
#     set(ANDROID_STL "c++_shared")
# endif()
