// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "msgpackIO.hpp"

#include <msgpack.hpp>

#include <sstream>
#include <string>

namespace aliceVision {
namespace sfmDataIO {



bool saveMsgPack(const sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag)
{
    return true;
}

bool loadMsgPack(sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag, bool incompleteViews = false,
              EViewIdMethod viewIdMethod = EViewIdMethod::METADATA, const std::string& viewIdRegex = "")
{
    return true;
}

} // namespace sfmDataIO
} // namespace aliceVision