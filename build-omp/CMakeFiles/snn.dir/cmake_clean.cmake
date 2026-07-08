file(REMOVE_RECURSE
  "libsnn.a"
  "libsnn.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CUDA)
  include(CMakeFiles/snn.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
