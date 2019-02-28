#pragma once

#import "phd/string.h"

#include "boost/filesystem.hpp"

namespace phd {

// Return the absolute path to a data file.
//
// This provides access to files from the 'data' attribute of a target in
// Bazel. Given a fully relative path to a data file,
// e.g. "phd/my/package/data", return the absolute path. The path must be
// relative to the bazel runfiles root, and begin with the name of the
// workspace.
//
boost::filesystem::path BazelDataPathOrDie(const string& path);

}  // namespace phd
