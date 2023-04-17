// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfmDataIO/viewIO.hpp>


#include <string>

namespace aliceVision {
namespace sfmDataIO {

/**
 * @brief Save an SfMData in a  msgpack file
 * @param[in] sfmData The input SfMData
 * @param[in] filename The filename
 * @param[in] partFlag The ESfMData save flag
 * @return true if completed
 */
bool saveMsgPack(const sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag);

/**
 * @brief Load a msgpack SfMData file.
 * @param[out] sfmData The output SfMData
 * @param[in] filename The filename
 * @param[in] partFlag The ESfMData load flag
 * @return true if completed
 */
bool loadMsgPack(sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag);

} // namespace sfmDataIO
} // namespace aliceVision
