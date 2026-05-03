cmake_minimum_required(VERSION 3.20)

project(face_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect highgui)

add_executable(face_cli
  src/main.cpp
  src/app_support.cpp
  src/face_pipeline.cpp
  src/face_matcher.cpp
)

add_executable(face_gui WIN32
  src/face_gui.cpp
  src/app_support.cpp
  src/face_pipeline.cpp
  src/face_matcher.cpp
)

target_include_directories(face_cli PRIVATE src)
target_include_directories(face_gui PRIVATE src)
target_link_libraries(face_cli PRIVATE ${OpenCV_LIBS})
target_link_libraries(face_gui PRIVATE ${OpenCV_LIBS} comdlg32 comctl32 shell32 ole32)

if(MINGW)
  target_compile_options(face_cli PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(face_gui PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MSVC)
  target_compile_options(face_cli PRIVATE /W4)
  target_compile_options(face_gui PRIVATE /W4)
endif()
