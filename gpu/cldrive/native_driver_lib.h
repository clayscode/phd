// TODO(cec):
//
#pragma once

#include "gpu/cldrive/proto/cldrive.pb.h"
#include "gpu/clinfo/libclinfo.h"

namespace gpu {
namespace cldrive {

void ProcessCldriveInstanceOrDie(CldriveInstance *instance);

} // namespace cldrive
} // namespace gpu
