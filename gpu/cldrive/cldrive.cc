// Main entry point for cldrive command line executable.
//
// Usage summary:
//   cldrive --srcs=<opencl_sources> --envs=<opencl_devices>
//       --gsize=<gsize> --lsize=<lsize> --output_format=(txt|pb|pbtxt)
//
// Run with `--help` argument to see full usage options.
//
// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/libcldrive.h"

#include "gpu/cldrive/csv_log.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "gpu/clinfo/libclinfo.h"

#include "phd/app.h"
#include "phd/logging.h"

#include "absl/strings/str_split.h"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "gflags/gflags.h"

#include <sstream>

namespace {

// Split a string into a vector of comma separated strings, e.g.
//     'a,b' -> 'a', 'b'
//     'ab' -> 'ab'
std::vector<string> SplitCommaSeparated(const string& str) {
  std::vector<absl::string_view> str_paths =
      absl::StrSplit(str, ',', absl::SkipEmpty());
  return std::vector<string>(str_paths.begin(), str_paths.end());
}

// Read file to string or abort.
string ReadFileOrDie(const string& path) {
  const boost::filesystem::path fs_path(path);
  CHECK(boost::filesystem::is_regular_file(fs_path)) << "Not a regular file: '"
                                                     << path << "'";
  boost::filesystem::ifstream istream(fs_path);
  CHECK(istream.is_open()) << "Failed to open: '" << path << "'";

  std::stringstream buffer;
  buffer << istream.rdbuf();
  return buffer.str();
}

}  // anonymous namespace

// Flag definitions ------------------------------------

DEFINE_string(srcs, "", "A comma separated list of OpenCL source files.");
static bool ValidateSrcs(const char* flagname, const string& value) {
  for (auto str_path : SplitCommaSeparated(value)) {
    // string str_path(str_path_view);
    boost::filesystem::path path(str_path);
    if (!boost::filesystem::is_regular_file(path)) {
      LOG(FATAL) << "File not found: " << value;
    }
  }

  return true;
}
DEFINE_validator(srcs, &ValidateSrcs);

DEFINE_string(envs, "",
              "A comma separated list of OpenCL devices to use. Use "
              "'--clinfo' argument to print a list of available devices.");
static bool ValidateEnvs(const char* flagname, const string& value) {
  for (auto env : SplitCommaSeparated(value)) {
    try {
      phd::gpu::clinfo::GetOpenClDevice(env);
    } catch (std::invalid_argument e) {
      LOG(ERROR) << "Available OpenCL environments:";
      auto devices = phd::gpu::clinfo::GetOpenClDevices();
      for (int i = 0; i < devices.device_size(); ++i) {
        LOG(ERROR) << "    " << devices.device(i).name();
      }
      LOG(FATAL) << "OpenCL environment '" << env << "' not found";
    }
  }
  return true;
}
DEFINE_validator(envs, &ValidateEnvs);

DEFINE_string(output_format, "csv",
              "The output format. One of: {csv,pb,pbtxt}.");
static bool ValidateOutputFormat(const char* flagname, const string& value) {
  if (value.compare("csv") && value.compare("pb") && value.compare("pbtxt")) {
    LOG(FATAL) << "Illegal value for --" << flagname << ". Must be one of: "
               << "{csv,pb,pbtxt}";
  }
  return true;
}
DEFINE_validator(output_format, &ValidateOutputFormat);

DEFINE_int32(gsize, 1024,
             "The global size to drive each kernel with. Buffers of this size "
             "are allocated and transferred for array arguments, and this many "
             "work items are instantiated.");
DEFINE_int32(lsize, 128, "The local (work group) size. Must be <= gsize.");
DEFINE_string(cl_build_opt, "", "Build options passed to clBuildProgram().");
DEFINE_int32(num_runs, 30, "The number of runs per kernel.");
DEFINE_bool(clinfo, false, "List the available devices and exit.");

// End flag definitions ------------------------------------

int main(int argc, char** argv) {
  phd::InitApp(&argc, &argv, "Drive arbitrary OpenCL kernels.");

  if (FLAGS_clinfo) {
    auto devices = phd::gpu::clinfo::GetOpenClDevices();
    for (int i = 0; i < devices.device_size(); ++i) {
      std::cout << devices.device(i).name() << std::endl;
    }
    return 0;
  }

  if (FLAGS_envs.empty()) {
    LOG(FATAL) << "Flag --envs must be set";
  }

  if (FLAGS_srcs.empty()) {
    LOG(FATAL) << "Flag --srcs must be set";
  }

  std::vector<::gpu::clinfo::OpenClDevice> devices;
  for (auto device_name : SplitCommaSeparated(FLAGS_envs)) {
    devices.push_back(
        phd::gpu::clinfo::GetOpenClDeviceProto(device_name).ValueOrDie());
  }

  // Print output headers.
  bool csv = !FLAGS_output_format.compare("csv");
  if (csv) {
    std::cout << gpu::cldrive::CsvLogHeader();
  } else if (!FLAGS_output_format.compare("pbtxt")) {
    std::cout << "# File: //gpu/cldrive/proto/cldrive.proto\n"
              << "# Proto: gpu.cldrive.CldriveInstances\n";
  }

  // Setup instance proto.
  gpu::cldrive::CldriveInstances instances;
  gpu::cldrive::CldriveInstance* instance = instances.add_instance();
  instance->set_build_opts(FLAGS_cl_build_opt);
  auto dp = instance->add_dynamic_params();
  dp->set_global_size_x(FLAGS_gsize);
  dp->set_local_size_x(FLAGS_lsize);
  instance->set_min_runs_per_kernel(FLAGS_num_runs);

  int instance_num = 0;
  for (auto path : SplitCommaSeparated(FLAGS_srcs)) {
    instance->set_opencl_src(ReadFileOrDie(path));

    for (size_t i = 0; i < devices.size(); ++i) {
      // Reset fields from previous loop iterations.
      instance->clear_outcome();
      instance->clear_kernel();

      *instance->mutable_device() = devices[i];

      gpu::cldrive::Cldrive(instance, instance_num).RunOrDie(csv);

      if (!FLAGS_output_format.compare("pb")) {
        instances.SerializeToOstream(&std::cout);
      } else if (!FLAGS_output_format.compare("pbtxt")) {
        std::cout << instances.DebugString();
      } else if (csv) {
        // Already handled
      } else {
        CHECK(false) << "unreachable!";
      }
    }

    ++instance_num;
  }

  return 0;
}
