
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(COMMAND git describe --tags
        OUTPUT_VARIABLE VERSION_STRING
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
    string(REGEX REPLACE "-" "." VERSION_STRING "${VERSION_STRING}")
    string(REGEX REPLACE ".g.*" "" VERSION_STRING "${VERSION_STRING}")

    set(tomopy_VERSION "${VERSION_STRING}")
    message(STATUS "tomopy version: ${tomopy_VERSION}")
else()
    set(tomopy_VERSION "0.0.0-unknown")
endif()
