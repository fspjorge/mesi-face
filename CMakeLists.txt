cmake_minimum_required(VERSION 3.20)

project(face_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect highgui)

set(STASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stasm)
set(STASM_SRC_DIR ${STASM_DIR}/stasm)

add_library(stasm STATIC
  ${STASM_SRC_DIR}/MOD_1/facedet.cpp
  ${STASM_SRC_DIR}/MOD_1/initasm.cpp
  ${STASM_SRC_DIR}/asm.cpp
  ${STASM_SRC_DIR}/classicdesc.cpp
  ${STASM_SRC_DIR}/convshape.cpp
  ${STASM_SRC_DIR}/err.cpp
  ${STASM_SRC_DIR}/eyedet.cpp
  ${STASM_SRC_DIR}/eyedist.cpp
  ${STASM_SRC_DIR}/faceroi.cpp
  ${STASM_SRC_DIR}/hat.cpp
  ${STASM_SRC_DIR}/hatdesc.cpp
  ${STASM_SRC_DIR}/landmarks.cpp
  ${STASM_SRC_DIR}/misc.cpp
  ${STASM_SRC_DIR}/pinstart.cpp
  ${STASM_SRC_DIR}/print.cpp
  ${STASM_SRC_DIR}/shape17.cpp
  ${STASM_SRC_DIR}/shapehacks.cpp
  ${STASM_SRC_DIR}/shapemod.cpp
  ${STASM_SRC_DIR}/startshape.cpp
  ${STASM_SRC_DIR}/stasm.cpp
  ${STASM_SRC_DIR}/stasm_lib.cpp
)

target_include_directories(stasm
  PUBLIC
    ${STASM_DIR}
    ${STASM_SRC_DIR}
    ${STASM_SRC_DIR}/MOD_1
)
target_compile_definitions(stasm
  PRIVATE
    _CRT_SECURE_NO_WARNINGS
    _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
)
target_link_libraries(stasm PUBLIC ${OpenCV_LIBS})

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
target_link_libraries(face_cli PRIVATE stasm ${OpenCV_LIBS})
target_link_libraries(face_gui PRIVATE stasm ${OpenCV_LIBS} comdlg32 comctl32 shell32 ole32)

if(MINGW)
  target_compile_options(stasm PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unknown-pragmas)
  target_compile_options(face_cli PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(face_gui PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MSVC)
  target_compile_options(stasm PRIVATE /W3)
  target_compile_options(face_cli PRIVATE /W4)
  target_compile_options(face_gui PRIVATE /W4)
endif()
