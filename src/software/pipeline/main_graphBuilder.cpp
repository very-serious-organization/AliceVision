// This file is part of the AliceVision project.
// Copyright (c) 2015 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/ProgressDisplay.hpp>
#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/feature/imageDescriberCommon.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/sfm/pipeline/pairwiseMatchesIO.hpp>

#include <aliceVision/camera/Pinhole.hpp>

#include <aliceVision/sfm/pipeline/RelativePoseInfo.hpp>
#include <aliceVision/multiview/triangulation/triangulationDLT.hpp>
#include <aliceVision/sfm/BundleAdjustmentCeres.hpp>
#include <aliceVision/geometry/rigidTransformation3D.hpp>

#include <aliceVision/sfm/liealgebra.hpp>

#include <cstdlib>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

using namespace aliceVision;

namespace po = boost::program_options;

bool getCovariance(Eigen::Matrix<double,6,6> & C, sfmData::SfMData & resultsfmData, const feature::FeaturesPerView & featuresPerView, const matching::MatchesPerDescType & viewMatches, const geometry::Pose3 & pose, std::shared_ptr<sfmData::View> firstView, std::shared_ptr<sfmData::View> secondView, std::shared_ptr<camera::Pinhole> firstPinhole, std::shared_ptr<camera::Pinhole> secondPinhole, bool inverseMatches)
{
    // Init structure
    const Mat34 Pfirst = firstPinhole->getProjectiveEquivalent(geometry::Pose3());
    const Mat34 Psecond = secondPinhole->getProjectiveEquivalent(pose);

    sfmData::SfMData tinyScene;
    tinyScene.views.insert(std::make_pair(firstView->getViewId(), firstView));
    tinyScene.views.insert(std::make_pair(secondView->getViewId(), secondView));

    tinyScene.intrinsics.insert(std::make_pair(firstView->getIntrinsicId(), firstPinhole));
    tinyScene.intrinsics.insert(std::make_pair(secondView->getIntrinsicId(), secondPinhole));

    tinyScene.setPose(*firstView, sfmData::CameraPose(geometry::Pose3()));
    tinyScene.setPose(*secondView, sfmData::CameraPose(pose));

    tinyScene.getPoses()[firstView->getPoseId()].lock();

    sfmData::Landmarks & landmarks = tinyScene.structure;

    size_t count = 0;
    for (auto matches : viewMatches)
    {
        const feature::PointFeatures& firstFeatures = featuresPerView.getFeatures(firstView->getViewId(), matches.first);
        const feature::PointFeatures& secondFeatures = featuresPerView.getFeatures(secondView->getViewId(), matches.first);

        for (auto match : matches.second)
        {
            IndexT firstMatchIndex = match._i;
            IndexT secondMatchIndex = match._j;

            if (inverseMatches)
            {
                firstMatchIndex = match._j;
                secondMatchIndex = match._i;
            }

            const feature::PointFeature& firstFeature = firstFeatures[firstMatchIndex];
            const feature::PointFeature& secondFeature = secondFeatures[secondMatchIndex];

            const Vec2 x1 = firstFeature.coords().cast<double>();
            const Vec2 x2 = secondFeature.coords().cast<double>();

            Vec3 X;
            multiview::TriangulateDLT(Pfirst, x1, Psecond, x2, &X);
            if (X(2) < 1e-6)
            {
                continue;
            }

            sfmData::Observations observations;
            observations[firstView->getViewId()] = sfmData::Observation(x1, firstMatchIndex, firstFeature.scale());
            observations[secondView->getViewId()] = sfmData::Observation(x2, secondMatchIndex, secondFeature.scale());


            sfmData::Landmark& newLandmark = landmarks[count];
            newLandmark.descType = matches.first;
            newLandmark.observations = observations;
            newLandmark.X = X;

            count++;
        }
    }
    
    sfm::BundleAdjustmentCeres::CeresOptions options(false, false);
    options.linearSolverType = ceres::DENSE_SCHUR;
    sfm::BundleAdjustmentCeres bundle(options);
    bool resultBundle = bundle.adjust(tinyScene, sfm::BundleAdjustment::REFINE_ROTATION | sfm::BundleAdjustment::REFINE_TRANSLATION | sfm::BundleAdjustment::REFINE_STRUCTURE);
    if (!resultBundle)
    {
        return false;
    }

    C = bundle.getCovariances()[secondView->getPoseId()];
    resultsfmData = tinyScene;

    return true;
}

Eigen::Matrix<double, 6, 6> getAdjoint(const Eigen::Matrix4d & SE3)
{
    Eigen::Matrix<double, 6, 6> ret = Eigen::Matrix<double, 6, 6>::Zero();

    const auto & R = SE3.block<3, 3>(0, 0);
    const auto & t = SE3.block<3, 1>(0, 3);

    ret.block<3, 3>(0, 0) = R;
    ret.block<3, 3>(0, 3) = SO3::skew(t) * R;
    ret.block<3, 3>(3, 3) = R;

    return ret;
}

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::vector<std::string> featuresFolders;
    std::vector<std::string> matchesFolders;
    std::string outputSfM;

    // user optional parameters
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    std::pair<std::string, std::string> initialPairString("", "");
    int maxNbMatches = 0;
    int minNbMatches = 0;
    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
      ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
        "SfMData file.")
      ("output,o", po::value<std::string>(&outputSfM)->required(),
        "Path to the output SfMData file.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
      ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(),
        "Path to folder(s) containing the extracted features.")
      ("matchesFolders,m", po::value<std::vector<std::string>>(&matchesFolders)->multitoken(),
        "Path to folder(s) in which computed matches are stored.")
      ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
        feature::EImageDescriberType_informations().c_str());

    CmdLine cmdline("AliceVision graphBuilder");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set maxThreads
    HardwareContext hwc = cmdline.getHardwareContext();
    omp_set_num_threads(hwc.getMaxThreads());

    std::mt19937 randomNumberGenerator;
    randomNumberGenerator.seed(randomSeed);

    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
        return EXIT_FAILURE;
    }

    // get imageDescriber type
    const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(describerTypesName);

    // features reading
    feature::FeaturesPerView featuresPerView;
    if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
    {
        ALICEVISION_LOG_ERROR("Invalid features.");
        return EXIT_FAILURE;
    }

    // matches reading
    matching::PairwiseMatches pairwiseMatches;
    if(!sfm::loadPairwiseMatches(pairwiseMatches, sfmData, matchesFolders, describerTypes, maxNbMatches, minNbMatches,
                                 true))
    {
        ALICEVISION_LOG_ERROR("Unable to load matches.");
        return EXIT_FAILURE;
    }

    auto progressDisplay = system::createConsoleProgressDisplay(pairwiseMatches.size(), std::cout, "\n- Essential matrices computation -\n" );
    
    struct LinkInfo
    {
        sfmData::SfMData sfmdata;
        Eigen::Matrix<double, 6, 6> C;
    };

    std::map<Pair, LinkInfo> links;

//#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pairwiseMatches.size(); ++i)
    {
        ++progressDisplay;
        {
            matching::PairwiseMatches::const_iterator iter(pairwiseMatches.begin());
            std::advance(iter, i);
            const auto& pairViewMatches(*iter);

            IndexT firstViewId = pairViewMatches.first.first;
            IndexT secondViewId = pairViewMatches.first.second;

            std::shared_ptr<sfmData::View> firstView = sfmData.getViews()[firstViewId];
            std::shared_ptr<sfmData::View> secondView = sfmData.getViews()[secondViewId];

            std::shared_ptr<camera::IntrinsicBase> firstIntrinsic = sfmData.getIntrinsicsharedPtr(firstView->getIntrinsicId());
            std::shared_ptr<camera::IntrinsicBase> secondIntrinsic = sfmData.getIntrinsicsharedPtr(secondView->getIntrinsicId());

            if (!firstIntrinsic || !secondIntrinsic)
            {
                ALICEVISION_LOG_ERROR("One of the view has no intrinsics defined");
                continue;
            }

            std::shared_ptr<camera::Pinhole> firstPinhole = std::dynamic_pointer_cast<camera::Pinhole>(firstIntrinsic);
            std::shared_ptr<camera::Pinhole> secondPinhole = std::dynamic_pointer_cast<camera::Pinhole>(secondIntrinsic);

            if (!firstPinhole || !secondPinhole)
            {
                ALICEVISION_LOG_ERROR("At least one of the view is not captured by a pinhole camera");
                continue;
            }

            size_t totalSize = 0;
            for (auto matches : pairViewMatches.second)
            {
                totalSize += matches.second.size();
            }

            Mat referencePoints(2, totalSize);
            Mat matchesPoints(2, totalSize);

            size_t pos = 0;
            for (auto matches : pairViewMatches.second)
            {
                const feature::PointFeatures& firstFeatures = featuresPerView.getFeatures(firstViewId, matches.first);
                const feature::PointFeatures& secondFeatures = featuresPerView.getFeatures(secondViewId, matches.first);

                for (auto match : matches.second)
                {
                    const feature::PointFeature& firstFeature = firstFeatures[match._i];
                    const feature::PointFeature& secondFeature = secondFeatures[match._j];

                    referencePoints(0, pos) = firstFeature.x();
                    referencePoints(1, pos) = firstFeature.y();
                    matchesPoints(0, pos) = secondFeature.x();
                    matchesPoints(1, pos) = secondFeature.y();

                    pos++;
                }
            }


            sfm::RelativePoseInfo poseInfo;

            bool res = sfm::robustRelativePose(firstPinhole->K(), secondPinhole->K(), referencePoints, matchesPoints, randomNumberGenerator, poseInfo, std::make_pair(firstPinhole->w(), firstPinhole->h()), std::make_pair(secondPinhole->w(), secondPinhole->h()));
            if (!res)
            {
                continue;
            }

            LinkInfo lsecond;
            if (!getCovariance(lsecond.C, lsecond.sfmdata, featuresPerView, pairViewMatches.second, poseInfo.relativePose, firstView, secondView, firstPinhole, secondPinhole, false))
            {
                continue;
            }

            LinkInfo lfirst;
            if (!getCovariance(lfirst.C, lfirst.sfmdata, featuresPerView, pairViewMatches.second, poseInfo.relativePose.inverse(), secondView, firstView, secondPinhole, firstPinhole, true))
            {
                continue;
            }

            links[std::make_pair(firstViewId, secondViewId)] = lsecond;
            links[std::make_pair(secondViewId, firstViewId)] = lfirst;
        }
    }

    typedef std::pair<Pair, Pair> edges;

    struct PairEdge
    {
        double scale;
        Mat3 R;
        Vec3 t;
    };
    
    for (auto ls : links)
    {
        IndexT ref = ls.first.second;

        sfmData::Landmarks landmarksRef = ls.second.sfmdata.getLandmarks();

        std::map<IndexT, Vec3> pointPerFeatureIndex;

        for (const auto & l : landmarksRef)
        {
            pointPerFeatureIndex[l.second.observations.at(ref).id_feat] = l.second.X;
        }

        std::vector<std::pair<Vec3, Vec3>> pts;

        for (auto lo : links)
        {
            IndexT other = lo.first.first;
            if (ref != other)
            {
                continue;
            }

            for (const auto& l : lo.second.sfmdata.getLandmarks())
            {
                IndexT featureIndex = l.second.observations.at(other).id_feat;
                auto it = pointPerFeatureIndex.find(featureIndex);
                if (it == pointPerFeatureIndex.end())
                {
                    continue;
                }

                Vec3 refPt = it->second;
                Vec3 curPt = l.second.X;

                pts.push_back({refPt, curPt});
            }
        }

        Mat refmat(3, pts.size());
        Mat curmat(3, pts.size());
        for (int i = 0; i < pts.size(); i++)
        {
            refmat(0, i) = pts[i].first.x();
            refmat(1, i) = pts[i].first.y();
            refmat(2, i) = pts[i].first.z();
            curmat(0, i) = pts[i].second.x();
            curmat(1, i) = pts[i].second.y();
            curmat(2, i) = pts[i].second.z();
        }

        PairEdge edge;
        aliceVision::track::TrackIdSet inliers;
        if (!geometry::ACRansac_FindRTS(refmat, curmat, randomNumberGenerator, edge.scale, edge.t, edge.R, inliers, true))
        {
            continue;
        }

        edges[std::make_pair()] = edge;
    }



    return EXIT_SUCCESS;
}

