// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "viewIO.hpp"

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/sfmData/uid.hpp>
#include <aliceVision/camera/camera.hpp>
#include <aliceVision/image/io.hpp>

#include <stdexcept>
#include <regex>

namespace fs = boost::filesystem;

namespace aliceVision {
namespace sfmDataIO {

void updateIncompleteView(sfmData::View& view, EViewIdMethod viewIdMethod, const std::string& viewIdRegex)
{
  // check if the view is complete
  if(view.getViewId() != UndefinedIndexT &&
     view.getIntrinsicId() != UndefinedIndexT &&
     view.getPoseId() == view.getViewId() &&
     view.getHeight() > 0 &&
     view.getWidth() >  0)
    return;

  int width, height;
  std::map<std::string, std::string> metadata;

  image::readImageMetadata(view.getImagePath(), width, height, metadata);

  view.setWidth(width);
  view.setHeight(height);

  // reset metadata
  if(view.getMetadata().empty())
    view.setMetadata(metadata);

  // Reset viewId
  if(view.getViewId() == UndefinedIndexT)
  {
    if(viewIdMethod == EViewIdMethod::FILENAME)
    {
      std::regex re;
      try
      {
        re = viewIdRegex;
      }
      catch(const std::regex_error& e)
      {
        throw std::invalid_argument("Invalid regex conversion, your regexfilename '" + viewIdRegex + "' may be invalid.");
      }

      // Get view image filename without extension
      const std::string filename = boost::filesystem::path(view.getImagePath()).stem().string();

      std::smatch match;
      std::regex_search(filename, match, re);
      if(match.size() == 2)
      {
          try
          {
            const IndexT id(std::stoul(match.str(1)));
            view.setViewId(id);
          }
          catch(std::invalid_argument& e)
          {
            ALICEVISION_LOG_ERROR("ViewId captured in the filename '" << filename << "' can't be converted to a number. "
                                  "The regex '" << viewIdRegex << "' is probably incorrect.");
            throw;
          }
      }
      else
      {
        ALICEVISION_LOG_ERROR("The Regex '" << viewIdRegex << "' must match a unique number in the filename " << filename << "' to be used as viewId.");
        throw std::invalid_argument("The Regex '" + viewIdRegex + "' must match a unique number in the filename " + filename + "' to be used as viewId.");
      }
    }
    else
    {
      // Use metadata
      view.setViewId(sfmData::computeViewUID(view));
    }
  }

  if(view.getPoseId() == UndefinedIndexT)
  {
    // check if the rig poseId id is defined
    if(view.isPartOfRig())
    {
      ALICEVISION_LOG_ERROR("Error: Can't find poseId for'" << fs::path(view.getImagePath()).filename().string() << "' marked as part of a rig." << std::endl);
      throw std::invalid_argument("Error: Can't find poseId for'" + fs::path(view.getImagePath()).filename().string() + "' marked as part of a rig.");
    }
    else
      view.setPoseId(view.getViewId());
  }
  else if((!view.isPartOfRig()) && (view.getPoseId() != view.getViewId()))
  {
    ALICEVISION_LOG_ERROR("Error: Bad poseId for image '" << fs::path(view.getImagePath()).filename().string() << "' (viewId should be equal to poseId)." << std::endl);
    throw std::invalid_argument("Error: Bad poseId for image '" + fs::path(view.getImagePath()).filename().string() + "'.");
  }
}

std::shared_ptr<camera::IntrinsicBase> getViewIntrinsic(
                    const sfmData::View& view, double mmFocalLength, double sensorWidth,
                    double defaultFocalLengthPx, double defaultFieldOfView,
                    camera::EINTRINSIC defaultIntrinsicType,
                    camera::EINTRINSIC allowedEintrinsics, 
                    double defaultPPx, double defaultPPy)
{
  // can't combine defaultFocalLengthPx and defaultFieldOfView
  assert(defaultFocalLengthPx < 0 || defaultFieldOfView < 0);

  // get view informations
  const std::string& cameraBrand = view.getMetadataMake();
  const std::string& cameraModel = view.getMetadataModel();
  const std::string& bodySerialNumber = view.getMetadataBodySerialNumber();
  const std::string& lensSerialNumber = view.getMetadataLensSerialNumber();

  double focalLengthIn35mm = mmFocalLength; // crop factor is apply later if sensor width is defined

  double pxFocalLength;
  bool hasFocalLengthInput = false;

  if(defaultFocalLengthPx > 0.0)
  {
    pxFocalLength = defaultFocalLengthPx;
  }

  if(defaultFieldOfView > 0.0)
  {
    const double focalRatio = 0.5 / std::tan(0.5 * degreeToRadian(defaultFieldOfView));
    pxFocalLength = focalRatio * std::max(view.getWidth(), view.getHeight());
  }

  camera::EINTRINSIC intrinsicType = defaultIntrinsicType;

  double ppx = view.getWidth() / 2.0;
  double ppy = view.getHeight() / 2.0;

  bool isResized = false;

  if(view.hasMetadata({"Exif:PixelXDimension", "PixelXDimension"}) && view.hasMetadata({"Exif:PixelYDimension", "PixelYDimension"})) // has dimension metadata
  {
    // check if the image is resized
    int exifWidth = std::stoi(view.getMetadata({"Exif:PixelXDimension", "PixelXDimension"}));
    int exifHeight = std::stoi(view.getMetadata({"Exif:PixelYDimension", "PixelXDimension"}));

    // if metadata is rotated
    if(exifWidth == view.getHeight() && exifHeight == view.getWidth())
      std::swap(exifWidth, exifHeight);


    if(exifWidth > 0 && exifHeight > 0 &&
       (exifWidth != view.getWidth() || exifHeight != view.getHeight()))
    {
      ALICEVISION_LOG_WARNING("Resized image detected: " << fs::path(view.getImagePath()).filename().string() << std::endl
                          << "\t- real image size: " <<  view.getWidth() << "x" <<  view.getHeight() << std::endl
                          << "\t- image size from exif metadata is: " << exifWidth << "x" << exifHeight << std::endl);
      isResized = true;
    }
  }
  else if(defaultPPx > 0.0 && defaultPPy > 0.0) // use default principal point
  {
    ppx = defaultPPx;
    ppy = defaultPPy;
  }

  // handle case where focal length (mm) is unset or false
  if(mmFocalLength <= 0.0)
  {
    ALICEVISION_LOG_WARNING("Image '" << fs::path(view.getImagePath()).filename().string() << "' focal length (in mm) metadata is missing." << std::endl
                             << "Can't compute focal length (px), use default." << std::endl);
  }
  else if(sensorWidth > 0.0)
  {
    // Retrieve the focal from the metadata in mm and convert to pixel.
    pxFocalLength = std::max(view.getWidth(), view.getHeight()) * mmFocalLength / sensorWidth;
    hasFocalLengthInput = true;

    //fieldOfView = radianToDegree(2.0 * std::atan(sensorWidth / (mmFocalLength * 2.0))); // [rectilinear] AFOV = 2 * arctan(sensorSize / (2 * focalLength))
    //fieldOfView = radianToDegree(4.0 * std::asin(sensorWidth / (mmFocalLength * 4.0))); // [fisheye] AFOV = 4 * arcsin(sensorSize / (4 * focalLength))

    focalLengthIn35mm *= 36.0 / sensorWidth; // multiply focal length by the crop factor
  }

  // choose intrinsic type
  if(cameraBrand == "Custom")
  {
    intrinsicType = camera::EINTRINSIC_stringToEnum(cameraModel);
  }
  /*
  // Warning: This resize heuristic is disabled as RAW images have a different size in metadata.
  else if(isResized)
  {
    // if the image has been resized, we assume that it has been undistorted
    // and we use a camera without lens distortion.
    intrinsicType = camera::PINHOLE_CAMERA;
  }
  */
  else if((focalLengthIn35mm > 0.0 && focalLengthIn35mm < 18.0) || (defaultFieldOfView > 100.0) && allowedEintrinsics & camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE)
  {
    // If the focal lens is short, the fisheye model should fit better.
      intrinsicType = camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE;
  }
  else if(intrinsicType == camera::EINTRINSIC::UNKNOWN)
  {
    // Choose a default camera model if no default type
    static const std::initializer_list<camera::EINTRINSIC> intrinsicsPriorities = {
        camera::EINTRINSIC::PINHOLE_CAMERA_RADIAL3,
        camera::EINTRINSIC::PINHOLE_CAMERA_BROWN,
        camera::EINTRINSIC::PINHOLE_CAMERA_RADIAL1,
        camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE,
        camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE1
    };

    for(const auto& e : intrinsicsPriorities)
    {
        if(allowedEintrinsics & e)
        {
            intrinsicType = e;
            break;
        }
    }
    // If still unassigned
    if(intrinsicType == camera::EINTRINSIC::UNKNOWN)
    {
        throw std::invalid_argument("No intrinsic value can be attributed !");
    }  
  }

  // create the desired intrinsic
  std::shared_ptr<camera::IntrinsicBase> intrinsic = camera::createPinholeIntrinsic(intrinsicType, view.getWidth(), view.getHeight(), pxFocalLength, ppx, ppy);
  if(hasFocalLengthInput)
    intrinsic->setInitialFocalLengthPix(pxFocalLength);

  // initialize distortion parameters
  switch(intrinsicType)
  {
      case camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE:
    {
      if(cameraBrand == "GoPro")
        intrinsic->updateFromParams({pxFocalLength, ppx, ppy, 0.0524, 0.0094, -0.0037, -0.0004});
      break;
    }
      case camera::EINTRINSIC::PINHOLE_CAMERA_FISHEYE1:
    {
      if(cameraBrand == "GoPro")
        intrinsic->updateFromParams({pxFocalLength, ppx, ppy, 1.04});
      break;
    }
    default: break;
  }

  // create serial number
  intrinsic->setSerialNumber(bodySerialNumber + lensSerialNumber);

  return intrinsic;
}

boost::filesystem::path viewPathFromFolders(const sfmData::View& view, const std::vector<std::string>& folders)
{
    boost::filesystem::path path = "";
    for(const std::string& folder : folders)
    {
        path = viewPathFromFolder(view, folder);
        if(!path.empty())
        {
            break;
        }
    }

    return path;
}

boost::filesystem::path viewPathFromFolder(const sfmData::View& view, const std::string& folder)
{
    const fs::recursive_directory_iterator end;
    const auto findIt = std::find_if(fs::recursive_directory_iterator(folder), end, 
        [&view](const fs::directory_entry& e)
        {
            const fs::path stem = e.path().stem();
            return (stem == std::to_string(view.getViewId()) || stem == fs::path(view.getImagePath()).stem());
        }
    );

    return (findIt != end) ? findIt->path() : "";
}

} // namespace sfmDataIO
} // namespace aliceVision
