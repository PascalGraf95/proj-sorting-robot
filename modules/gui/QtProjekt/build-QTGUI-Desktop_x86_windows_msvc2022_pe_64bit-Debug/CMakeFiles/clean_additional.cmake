# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\SortingGU_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\SortingGU_autogen.dir\\ParseCache.txt"
  "SortingGU_autogen"
  )
endif()
