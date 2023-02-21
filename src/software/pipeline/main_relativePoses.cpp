// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfmDataIO/jsonIO.hpp>
#include <aliceVision/sfm/sfm.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>
#include <aliceVision/system/ProgressDisplay.hpp>
#include <aliceVision/geometry/rigidTransformation3D.hpp>
#include <aliceVision/multiview/triangulation/triangulationDLT.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <cstdlib>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

using namespace aliceVision;
namespace po = boost::program_options;

bool computeMiniSFm(std::shared_ptr<sfmData::SfMData> & resultsfmData, const feature::FeaturesPerView & featuresPerView, const matching::MatchesPerDescType & viewMatches, const geometry::Pose3 & pose, std::shared_ptr<sfmData::View> firstView, std::shared_ptr<sfmData::View> secondView, std::shared_ptr<camera::Pinhole> firstPinhole, std::shared_ptr<camera::Pinhole> secondPinhole, bool inverseMatches)
{
    // Init structure
    const Mat34 Pfirst = firstPinhole->getProjectiveEquivalent(geometry::Pose3());
    const Mat34 Psecond = secondPinhole->getProjectiveEquivalent(pose);

    resultsfmData->views.insert(std::make_pair(firstView->getViewId(), firstView));
    resultsfmData->views.insert(std::make_pair(secondView->getViewId(), secondView));

    resultsfmData->intrinsics.insert(std::make_pair(firstView->getIntrinsicId(), firstPinhole));
    resultsfmData->intrinsics.insert(std::make_pair(secondView->getIntrinsicId(), secondPinhole));

    resultsfmData->setPose(*firstView, sfmData::CameraPose(geometry::Pose3()));
    resultsfmData->setPose(*secondView, sfmData::CameraPose(pose));

    //resultsfmData->getPoses()[firstView->getPoseId()].lock();

    sfmData::Landmarks & landmarks = resultsfmData->structure;

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

            //Check chierality
            Vec3 X;
            multiview::TriangulateDLT(Pfirst, x1, Psecond, x2, &X);
            if (X(2) < 1e-6)
            {
                continue;
            }

            Vec3 proj = (Psecond * X.homogeneous());


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

    
    sfm::BundleAdjustmentCeres::CeresOptions options(false, true);
    sfm::BundleAdjustmentCeres bundle(options);
    bool resultBundle = bundle.adjust(*resultsfmData, sfm::BundleAdjustment::REFINE_ROTATION | sfm::BundleAdjustment::REFINE_TRANSLATION | sfm::BundleAdjustment::REFINE_STRUCTURE);
    if (!resultBundle)
    {
        return false;
    }

    if (bundle.getStatistics().RMSEfinal > 1.0)
    {
        return false;
    }

    if (!bundle.computePoseUncertainty(*resultsfmData, sfm::BundleAdjustment::REFINE_ROTATION | sfm::BundleAdjustment::REFINE_TRANSLATION | sfm::BundleAdjustment::REFINE_STRUCTURE, firstView->getPoseId()))
    {
        return false;
    }

    return true;
}

bool build(std::shared_ptr<sfmData::SfMData> & pairSfmData, const sfmData::SfMData & sfmData, std::mt19937 & randomNumberGenerator, feature::FeaturesPerView featuresPerView, const matching::MatchesPerDescType & pairViewMatches, IndexT firstViewId, IndexT secondViewId)
{
    std::shared_ptr<sfmData::View> firstView = sfmData.getViews().at(firstViewId);
    std::shared_ptr<sfmData::View> secondView = sfmData.getViews().at(secondViewId);
    std::shared_ptr<camera::IntrinsicBase> firstIntrinsic = sfmData.getIntrinsicsharedPtr(firstView->getIntrinsicId());
    std::shared_ptr<camera::IntrinsicBase> secondIntrinsic = sfmData.getIntrinsicsharedPtr(secondView->getIntrinsicId());
    if (!firstIntrinsic || !secondIntrinsic)
    {
        ALICEVISION_LOG_ERROR("One of the view has no intrinsics defined");
        return false;
    }

    std::shared_ptr<camera::Pinhole> firstPinhole = std::dynamic_pointer_cast<camera::Pinhole>(firstIntrinsic);
    std::shared_ptr<camera::Pinhole> secondPinhole = std::dynamic_pointer_cast<camera::Pinhole>(secondIntrinsic);
    if (!firstPinhole || !secondPinhole)
    {
        ALICEVISION_LOG_ERROR("At least one of the view is not captured by a pinhole camera");
        return false;
    }

    size_t totalSize = 0;
    for (auto matches : pairViewMatches)
    {
        totalSize += matches.second.size();
    }

    Mat referencePoints(2, totalSize);
    Mat matchesPoints(2, totalSize);

    size_t pos = 0;
    for (auto matches : pairViewMatches)
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

    //First estimate pose
    sfm::RelativePoseInfo poseInfo;
    bool res = sfm::robustRelativePose(firstPinhole->K(), secondPinhole->K(), referencePoints, matchesPoints, randomNumberGenerator, poseInfo, std::make_pair(firstPinhole->w(), firstPinhole->h()), std::make_pair(secondPinhole->w(), secondPinhole->h()));
    if (!res)
    {
        return false;
    }

    std::shared_ptr<sfmData::SfMData> localSfmData = std::make_shared<sfmData::SfMData>();
    if (!computeMiniSFm(localSfmData, featuresPerView, pairViewMatches, poseInfo.relativePose, firstView, secondView, firstPinhole, secondPinhole, false))
    {
        return false;
    }

    pairSfmData = localSfmData;
}  

void buildTriplet(const std::map<std::pair<IndexT, IndexT>, std::shared_ptr<sfmData::SfMData>> &  builtSfmDatas, std::mt19937& randomNumberGenerator, IndexT start, IndexT middle, IndexT end)
{
    Pair p1(start, middle);
    Pair p2(middle, end);

    bool error = false;
    #pragma omp critical
    {
        if (builtSfmDatas.find(p1) == builtSfmDatas.end())
        {
            error = true;
        }
        else if (builtSfmDatas.find(p2) == builtSfmDatas.end())
        {
            error = true;
        }
    }

    if (error)
    {
        return;
    }


    const std::shared_ptr<sfmData::SfMData> sfmData1 = builtSfmDatas.at(p1);
    const std::shared_ptr<sfmData::SfMData> sfmData2 = builtSfmDatas.at(p2);

    if (sfmData1 == nullptr)
    {
        return;
    }

    if (sfmData2 == nullptr)
    {
        return;
    }

    std::unordered_map<IndexT, IndexT> indexLandmarks;

    //Search matches
    for (const auto & l : sfmData1->getLandmarks())
    {
        for (const auto& o : l.second.observations)
        {
            indexLandmarks[o.second.id_feat] = l.first;
        }
    }

    std::set<Pair> landmarkPairs;
    for (const auto& l : sfmData2->getLandmarks())
    {
        for (const auto& o : l.second.observations)
        {
            IndexT feat = o.second.id_feat;

            auto it = indexLandmarks.find(feat);
            if (it == indexLandmarks.end())
            {
                continue;
            }

            landmarkPairs.insert(Pair(it->second, l.first));
        }
    }

    Mat refmat(3, landmarkPairs.size());
    Mat curmat(3, landmarkPairs.size());

    Vec3 sum1, sum2;
    for (auto p : landmarkPairs)
    {
        const sfmData::Landmark& l1 = sfmData1->getLandmarks().at(p.first);
        const sfmData::Landmark& l2 = sfmData2->getLandmarks().at(p.second);

        sum1 += l1.X;
        sum2 += l2.X;
    }

    Vec3 mean1 = sum1 / double(landmarkPairs.size());
    Vec3 mean2 = sum2 / double(landmarkPairs.size());

    int idx = 0;
    for (auto p : landmarkPairs)
    {
        const sfmData::Landmark& l1 = sfmData1->getLandmarks().at(p.first);
        const sfmData::Landmark& l2 = sfmData2->getLandmarks().at(p.second);

        Vec3 l1X = (l1.X - mean1).normalized();
        Vec3 l2X = (l2.X - mean2).normalized();

        refmat(0, idx) = l1X.x();
        refmat(1, idx) = l1X.y();
        refmat(2, idx) = l1X.z();
        curmat(0, idx) = l2X.x();
        curmat(1, idx) = l2X.y();
        curmat(2, idx) = l2X.z();

        idx++;
    }

    aliceVision::track::TrackIdSet inliers;
    double S;
    Vec3 t;
    Mat3 R;
    if (!geometry::ACRansac_FindRTS(refmat, curmat, randomNumberGenerator, S, t, R, inliers, true))
    {
        return;
    }

    if (inliers.size() > 0) std::cout << start << " " << middle << " " << end << " "  << inliers.size() << std::endl;
}

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::vector<std::string> featuresFolders;
    std::vector<std::string> matchesFolders;
    std::string outputSfMpath;
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);

    int randomSeed = std::mt19937::default_seed;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(), "SfMData file.")
        ("output,o", po::value<std::string>(&outputSfMpath)->required(), "Path to the output SfMData files folder.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(), "Path to folder(s) containing the extracted features.")
        ("matchesFolders,m", po::value<std::vector<std::string>>(&matchesFolders)->multitoken(), "Path to folder(s) in which computed matches are stored.")
        ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName), feature::EImageDescriberType_informations().c_str());

    CmdLine cmdline("AliceVision relativePoses");

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
    if(!sfm::loadPairwiseMatches(pairwiseMatches, sfmData, matchesFolders, describerTypes, 0, 0, true))
    {
        ALICEVISION_LOG_ERROR("Unable to load matches.");
        return EXIT_FAILURE;
    }

//#pragma omp parallel for
    for (int id = 0; id < pairwiseMatches.size(); id++)
    {
        auto itmatch = pairwiseMatches.begin();
        std::advance(itmatch, id);

        
        std::shared_ptr<sfmData::SfMData> pairSfmData;
        if (!build(pairSfmData, sfmData, randomNumberGenerator, featuresPerView, itmatch->second, itmatch->first.first, itmatch->first.second))
        {
            continue;
        }


    }

    /*typedef boost::property<boost::vertex_name_t, IndexT> vertex_property_t;
    typedef boost::property<boost::edge_weight_t, double> edge_property_t;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_property_t, edge_property_t> graph_t;
    typedef graph_t::vertex_descriptor vertex_t;
    typedef graph_t::edge_descriptor edge_t;
    typedef boost::graph_traits<graph_t>::edge_iterator edge_iter;

    graph_t g; 
    boost::property_map<graph_t, boost::vertex_name_t>::type vertName = boost::get(boost::vertex_name, g);

    std::map<IndexT, vertex_t> nodes;
    for (const auto& pv : sfmData.getViews())
    {
        nodes[pv.first] = boost::add_vertex(pv.first, g);
    }

    for (const auto& match : pairwiseMatches)
    {
        const vertex_t & v1 = nodes[match.first.first];
        const vertex_t & v2 = nodes[match.first.second];
        boost::add_edge(v1, v2, 0.0, g);
        boost::add_edge(v2, v1, 0.1, g);
    }

    std::map<std::pair<IndexT, IndexT>, std::shared_ptr<sfmData::SfMData>> builtSfmDatas;

    auto edges = boost::edges(g);
    
//#pragma omp parallel for
    for (int id = 0; id < boost::num_edges(g); id++)
    {    
        auto eit = edges.first;
        std::advance(eit, id);

        vertex_t start = boost::source(*eit, g);
        vertex_t end = boost::target(*eit, g);

        if (vertName[start] != 1718128794) continue;

        build(builtSfmDatas, sfmData, randomNumberGenerator, featuresPerView, pairwiseMatches, vertName[start], vertName[end]);

        auto other_edges = boost::out_edges(end, g);


        for (auto oeit = other_edges.first; oeit != other_edges.second; ++oeit)
        {
            vertex_t other_end = boost::target(*oeit, g);
            if (other_end == start)
            {
                continue;
            }

            if (vertName[other_end] != 1060277544) continue;

            build(builtSfmDatas, sfmData, randomNumberGenerator, featuresPerView, pairwiseMatches, vertName[end], vertName[other_end]);

            buildTriplet(builtSfmDatas, randomNumberGenerator, vertName[start], vertName[end], vertName[other_end]);
        }
    }*/

    return EXIT_SUCCESS;
}
