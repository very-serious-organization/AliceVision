// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfmDataIO/sceneSample.hpp>

#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/main.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR   0

using namespace aliceVision;

/*
 * This program is used to load the sift descriptors from a list of files and create a vocabulary tree
 */
int aliceVision_main(int argc, char** argv)
{   
    aliceVision::sfmData::SfMData output;
    sfmDataIO::generateLargeScene(output);

    sfmDataIO::Save(output, "test.sfm", sfmDataIO::ESfMData::ALL);
    sfmDataIO::Save(output, "test.abc", sfmDataIO::ESfMData::ALL);
    sfmDataIO::Save(output, "test.msg", sfmDataIO::ESfMData::ALL);

    return EXIT_SUCCESS;
}
