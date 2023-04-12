// This file is part of the AliceVision project.
// Copyright (c) 2015 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/cmdline.hpp>

#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/matching/matcherType.hpp>
#include <aliceVision/matching/RegionsMatcher.hpp>
#include <aliceVision/matchingImageCollection/GeometricFilterMatrix_F_AC.hpp>
#include <aliceVision/robustEstimation/estimators.hpp>
#include <aliceVision/matching/io.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <mutex>
#include <cstdlib>
#include <fstream>
#include <cctype>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace aliceVision;

const feature::MapRegionsPerDesc & getOrLoadRegions(
                                            std::map<IndexT, feature::MapRegionsPerDesc> & map,
                                            const std::vector<std::string>& folders,
                                            const std::vector<std::unique_ptr<feature::ImageDescriber>> & describers, 
                                            const IndexT viewId)
{

    if(map.find(viewId) == map.end())
    {
        feature::MapRegionsPerDesc perDesc;
        for (auto & imageDescriber : describers)
        {
            std::unique_ptr<feature::Regions> vreg = sfm::loadRegions(folders, viewId, *imageDescriber);
            perDesc[imageDescriber->getDescriberType()] = std::move(vreg);            
        }

        map[viewId] = std::move(perDesc);
    }

    return map[viewId];
}

/// Compute corresponding features between a series of views:
/// - Load view images description (regions: features & descriptors)
/// - Compute putative local feature matches (descriptors matching)
/// - Compute geometric coherent feature matches (robust model estimation from putative matches)
/// - Export computed data
int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::string matchesFolder;
    std::vector<std::string> featuresFolders;

    // user optional parameters
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);
    int rangeStart = -1;
    int rangeSize = 0;
    const std::string fileExtension = "txt";
    int randomSeed = std::mt19937::default_seed;

    

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()("input,i", po::value<std::string>(&sfmDataFilename)->required(),
                                 "SfMData file.")("output,o", po::value<std::string>(&matchesFolder)->required(),
                                                  "Path to a folder in which computed matches will be stored.")(
        "featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken()->required(),
        "Path to folder(s) containing the extracted features.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()("describerTypes,d",
                                 po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
                                 feature::EImageDescriberType_informations().c_str())(
        "rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart),
        "Range image index start.")("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize), "Range size.")(
        "randomSeed", po::value<int>(&randomSeed)->default_value(randomSeed),
        "This seed value will generate a sequence using a linear random generator. Set -1 to use a random seed.");

    CmdLine cmdline("This program computes corresponding features between a series of views.\n"
                    "AliceVision sequentialFeatureMatching");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if(!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    std::mt19937 randomNumberGenerator(randomSeed == -1 ? std::random_device()() : randomSeed);

    // check and set input options
    if(matchesFolder.empty() || !fs::is_directory(matchesFolder))
    {
        ALICEVISION_LOG_ERROR("Invalid output matches folder: " + matchesFolder);
        return EXIT_FAILURE;
    }

    if(describerTypesName.empty())
    {
        ALICEVISION_LOG_ERROR("Empty option: --describerMethods");
        return EXIT_FAILURE;
    }

    sfmData::SfMData sfmData;
    if(!sfmDataIO::Load(sfmData, sfmDataFilename,
                        sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS | sfmDataIO::EXTRINSICS)))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" << sfmDataFilename << "' cannot be read.");
        return EXIT_FAILURE;
    }

    std::vector<Pair> viewsOrderedByFrameId;
    for(const auto& pv : sfmData.getViews())
    {
        IndexT frameId = pv.second->getFrameId();
        if(frameId == UndefinedIndexT)
        {
            ALICEVISION_LOG_ERROR("A view has no frameId.");
            return EXIT_FAILURE;
        }

        viewsOrderedByFrameId.push_back({frameId, pv.first});
    }

    std::sort(viewsOrderedByFrameId.begin(), viewsOrderedByFrameId.end(),
              [](const Pair& p1, const Pair& p2) { return p1.first < p2.first; });

    const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(describerTypesName);



    // for all describer types used
    std::map<IndexT, feature::MapRegionsPerDesc> loadedRegions;

    std::vector<std::unique_ptr<feature::ImageDescriber>> imageDescribers;
    for(feature::EImageDescriberType describerType: describerTypes)
    {
        imageDescribers.push_back(createImageDescriber(describerType));
    }

    if (rangeStart + rangeSize > viewsOrderedByFrameId.size())
    {
        rangeSize = viewsOrderedByFrameId.size() - rangeStart;
    }

    

    matching::PairwiseMatches finalMatches;

    // Loop over reference images
    std::mutex mutex;
    #pragma omp parallel for
    for(int idx = rangeStart; idx < rangeStart + rangeSize; idx++)
    {
        IndexT referenceViewIndex = viewsOrderedByFrameId[idx].second;

        mutex.lock();
        const feature::MapRegionsPerDesc & referenceRegions = getOrLoadRegions(loadedRegions, featuresFolders, imageDescribers, referenceViewIndex);
        mutex.unlock();

        matchingImageCollection::GeometricFilterMatrix_F_AC geometricFilter(std::numeric_limits<double>::infinity(), 2048, robustEstimation::ERobustEstimator::ACRANSAC);

        int lastIndex = 0;

        for (int nextIdx = idx + 1; nextIdx < viewsOrderedByFrameId.size(); nextIdx++)
        {
            lastIndex = nextIdx;
            IndexT nextViewIndex = viewsOrderedByFrameId[nextIdx].second;

            mutex.lock();
            const feature::MapRegionsPerDesc & nextRegions = getOrLoadRegions(loadedRegions, featuresFolders, imageDescribers, nextViewIndex);
            mutex.unlock();

            const sfmData::View &viewI = sfmData.getView(referenceViewIndex);
            const sfmData::View &viewJ = sfmData.getView(nextViewIndex);
            const std::shared_ptr<camera::IntrinsicBase> camI = sfmData.getIntrinsicsharedPtr(viewI.getIntrinsicId());
            const std::shared_ptr<camera::IntrinsicBase> camJ = sfmData.getIntrinsicsharedPtr(viewJ.getIntrinsicId());

            std::pair<std::size_t, std::size_t> sizeI = std::make_pair(camI->w(), camI->h());
            std::pair<std::size_t, std::size_t> sizeJ = std::make_pair(camJ->w(), camJ->h());

            matching::MatchesPerDescType matches;
            for (const auto& imageDescriber : imageDescribers)
            {
                const feature::Regions & regionsRef = *referenceRegions.at(imageDescriber->getDescriberType());
                const feature::Regions & regionsNext = *nextRegions.at(imageDescriber->getDescriberType());

                matching::RegionsDatabaseMatcher matcher(randomNumberGenerator, matching::EMatcherType::ANN_L2, regionsRef);
              
                matching::IndMatches putativesMatches;
                matcher.Match(0.8, regionsNext, putativesMatches); 

                matches[imageDescriber->getDescriberType()] = putativesMatches;           
            }     


            matching::MatchesPerDescType geoMatches;
            EstimationStatus status = geometricFilter.geometricEstimation(referenceRegions, nextRegions, camI.get(), camJ.get(), sizeI, sizeJ, matches, randomNumberGenerator, geoMatches);
            if (!status.isValid)
            {
                break;
            }

            if (geoMatches.getNbAllMatches() < 32)
            {
                break;
            }

            Pair vpair = std::make_pair(referenceViewIndex, nextViewIndex);

            mutex.lock();
            finalMatches[vpair] = geoMatches;
            mutex.unlock();
        }

        std::cout << idx << " --> " << lastIndex << std::endl;
    }

    const std::string filePrefix = rangeSize > 0 ? std::to_string(rangeStart/rangeSize) + "." : "";
    matching::Save(finalMatches, matchesFolder, fileExtension, false, filePrefix);

    return EXIT_SUCCESS;
}
